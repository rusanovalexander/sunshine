"""
=================================================
Stage 2: Multi-Pass Evidence-Based Extraction (v2)
=================================================
Key Improvements:
1. Extract fields in logical groups (5-8 fields at a time)
2. Require evidence quotes for every value
3. Use retrieval instead of lossy summarization
4. JSON output for robust parsing
5. Self-verification loop
6. Facility detection first, then per-facility extraction
"""

import os
import gc
import re
import json
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# GPU Optimization
from .gpu_optimizer import (
    get_gpu_memory_stats, log_gpu_memory, aggressive_memory_cleanup,
    get_model_manager, load_llm_optimized, GPUMemoryMonitor,
    BatchInferenceEngine, get_optimized_generation_config,
    get_memory_efficient_generation_kwargs, gpu_memory_context
)

# Company Document Consolidation
from .consolidate import (
    consolidate_company_documents, save_consolidated_document,
    get_all_companies, create_extraction_plan, get_chunks_for_extraction,
    merge_extraction_results, ConsolidatedDocument
)

from .config import (
    MODEL_PATH, PREPROCESSED_DATA_DIR, CHUNKS_DIR, EXTRACTION_DIR, OUTPUT_CSV,
    FIELD_GROUPS, ALL_FIELDS, EXTRACTABLE_FIELDS,
    EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE,
    FACILITY_DETECTION_PROMPT, VERIFICATION_PROMPT,
    MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY,
    MAX_CHUNKS_PER_FIELD_GROUP, FIELD_GROUP_MAX_TOKENS,
    FIELD_GROUP_SYSTEM_PROMPTS
)
from .retriever import BM25Retriever, create_retriever_from_chunks, retrieve_for_field_group, load_embedding_model

# Few-shot prompting
try:
    from examples.few_shot_template import create_enhanced_extraction_prompt
    FEW_SHOT_AVAILABLE = True
except ImportError:
    FEW_SHOT_AVAILABLE = False

# Output normalization
from .normalize import normalize_field_value

# =====================================================================
# LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Extract - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class ExtractedValue:
    """Single extracted field value with evidence."""
    value: str
    evidence: str
    confidence: str  # HIGH, MEDIUM, LOW
    source_chunk: int = -1


@dataclass
class FacilityData:
    """Data for a single facility/tranche."""
    facility_name: str
    facility_type: str
    fields: Dict[str, ExtractedValue]
    raw_extractions: Dict[str, Dict]  # group_name -> raw extraction results


@dataclass
class DocumentExtraction:
    """Complete extraction results for a document."""
    company: str
    source_file: str
    facilities: List[FacilityData]
    document_summary: str
    extraction_timestamp: str
    confidence_score: float


# =====================================================================
# MODEL INITIALIZATION (Optimized)
# =====================================================================

def initialize_model(model_path: str, use_flash_attention: bool = False):
    """Load Qwen3-14B with optimized 4-bit quantization for A100 20GB."""
    logger.info(f"Loading model: {model_path}")
    log_gpu_memory("Before model load: ")
    
    try:
        model, tokenizer = load_llm_optimized(
            model_path,
            use_flash_attention=use_flash_attention,
            max_memory_gb=14.0  # Leave ~7GB headroom for KV cache on 21GB MIG
        )
        
        logger.info("Model loaded successfully")
        log_gpu_memory("After model load: ")
        return model, tokenizer
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return None, None


def llm_generate(prompt_messages: List[Dict], model, tokenizer, 
                 max_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate LLM response with memory-efficient settings."""
    try:
        text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Get memory-efficient generation kwargs
        gen_kwargs = get_memory_efficient_generation_kwargs()
        gen_kwargs["max_new_tokens"] = max_tokens
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        
        # Check available memory and adjust if needed
        stats = get_gpu_memory_stats()
        if stats and stats.free_gb < 3.0:
            logger.warning(f"Low GPU memory ({stats.free_gb:.2f}GB free), reducing max_tokens")
            gen_kwargs["max_new_tokens"] = min(max_tokens, 1024)
            aggressive_memory_cleanup()
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Cleanup
        del inputs, outputs
        torch.cuda.empty_cache()
        
        return response
    except torch.cuda.OutOfMemoryError:
        logger.error("OOM during generation, attempting recovery...")
        aggressive_memory_cleanup()
        
        # Retry with minimal tokens
        try:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            del inputs, outputs
            torch.cuda.empty_cache()
            return response
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return ""
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return ""


def parse_json_response(response: str) -> Optional[Dict]:
    """Robustly parse JSON from LLM response."""
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON block in response
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response)
        for match in matches:
            try:
                # Handle both string match and the full match
                json_str = match if isinstance(match, str) else match[0]
                if not json_str.startswith('{'):
                    json_str = '{' + json_str.split('{', 1)[-1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    logger.warning("Failed to parse JSON from response")
    return None


# =====================================================================
# FACILITY DETECTION
# =====================================================================

def detect_facilities(document_text: str, model, tokenizer,
                     retriever: BM25Retriever = None) -> List[Dict]:
    """
    First pass: Detect all distinct facilities/tranches in the document.
    This helps us know if we need per-facility extraction.

    Enhanced with multi-section sampling: instead of only reading the
    first 15k chars, we also pull relevant sections from schedules
    and appendices using BM25 retrieval (CPU-only).
    """
    # Start with the beginning of the document (usually has facility overview)
    text_parts = [document_text[:10000]]

    # Use BM25 to find sections mentioning facilities/tranches/schedules
    if retriever:
        facility_keywords = (
            "facility tranche schedule term loan revolving RCF "
            "letter of credit commitment amount aggregate"
        )
        bm25_results = retriever.search(facility_keywords, top_k=5)
        for _, _, chunk_text in bm25_results:
            # Avoid duplicating content from the beginning
            if chunk_text[:200] not in document_text[:10000]:
                text_parts.append(chunk_text)

    # Also grab from the end of the document (schedules are often at the end)
    if len(document_text) > 20000:
        text_parts.append(document_text[-5000:])

    # Combine, keeping within ~15k char budget for the LLM
    combined = "\n\n---\n\n".join(text_parts)
    if len(combined) > 15000:
        combined = combined[:15000]

    prompt = FACILITY_DETECTION_PROMPT.format(document_text=combined)

    messages = [
        {"role": "system", "content": "You are a financial analyst. Respond only with valid JSON."},
        {"role": "user", "content": prompt}
    ]

    response = llm_generate(messages, model, tokenizer, max_tokens=2048)
    result = parse_json_response(response)

    if result and 'facilities' in result:
        logger.info(f"Detected {result.get('total_facilities', 0)} facilities")
        return result['facilities']

    # Default: single unnamed facility
    return [{"name": "Primary Facility", "type": "Unknown", "amount": "N/A"}]


# =====================================================================
# FIELD GROUP EXTRACTION
# =====================================================================

def extract_field_group(group_name: str, group_config: Dict,
                       relevant_chunks: List[str],
                       facility_context: str,
                       model, tokenizer) -> Dict:
    """
    Extract a single group of fields from relevant chunks.
    Returns structured extraction with evidence.

    Enhanced with:
    - Per-field-group few-shot examples for better accuracy
    - Adaptive max_new_tokens to save KV cache memory on A100 20GB
    """
    fields = group_config['fields']

    # Build field descriptions
    field_desc = []
    for field_name, field_help in fields:
        field_desc.append(f"- **{field_name}**: {field_help}")
    fields_description = '\n'.join(field_desc)

    # Combine relevant chunks
    document_text = '\n\n---\n\n'.join(relevant_chunks)

    # Add facility context if we have multiple facilities
    if facility_context:
        document_text = f"[CONTEXT: Extracting data for {facility_context}]\n\n{document_text}"

    # Build prompt
    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        fields_description=fields_description,
        document_text=document_text
    )

    # Enhance with per-group few-shot example
    if FEW_SHOT_AVAILABLE:
        user_prompt = create_enhanced_extraction_prompt(user_prompt, group_name=group_name)

    # Use group-specific system prompt if available, otherwise generic
    system_prompt = FIELD_GROUP_SYSTEM_PROMPTS.get(group_name, EXTRACTION_SYSTEM_PROMPT)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Use adaptive max_new_tokens per field group to save KV cache memory
    group_max_tokens = FIELD_GROUP_MAX_TOKENS.get(group_name, MAX_NEW_TOKENS)

    response = llm_generate(messages, model, tokenizer, max_tokens=group_max_tokens)
    result = parse_json_response(response)

    if result:
        return result

    # Return empty structure if parsing failed
    return {
        "fields": {name: {"value": "EXTRACTION_ERROR", "evidence": "", "confidence": "LOW"}
                  for name, _ in fields},
        "notes": "Failed to parse extraction response"
    }


def extract_all_field_groups(retriever: BM25Retriever, 
                             facility_context: str,
                             model, tokenizer) -> Dict[str, Dict]:
    """
    Multi-pass extraction: Extract each field group separately.
    """
    all_extractions = {}
    
    for group_name, group_config in FIELD_GROUPS.items():
        logger.info(f"  Extracting: {group_config['name']}")
        
        # Get field names for this group
        field_names = [name for name, _ in group_config['fields']]
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_for_field_group(
            retriever,
            group_config['keywords'],
            field_names,
            top_k=MAX_CHUNKS_PER_FIELD_GROUP
        )
        
        if not relevant_chunks:
            logger.warning(f"    No relevant chunks found for {group_name}")
            relevant_chunks = ["[No relevant content found]"]
        
        # Extract
        extraction = extract_field_group(
            group_name, group_config, relevant_chunks, facility_context, model, tokenizer
        )
        
        all_extractions[group_name] = extraction
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_extractions


# =====================================================================
# VERIFICATION PASS
# =====================================================================

def verify_extraction(extractions: Dict, document_sample: str, 
                     model, tokenizer) -> Dict:
    """
    Verification pass: Check extracted values against source.
    Focus on high-value fields that are commonly wrong.
    """
    # Collect all extracted values
    extracted_summary = []
    for group_name, group_data in extractions.items():
        if 'fields' not in group_data:
            continue
        for field_name, field_data in group_data['fields'].items():
            if isinstance(field_data, dict):
                value = field_data.get('value', 'N/A')
                extracted_summary.append(f"- {field_name}: {value}")
    
    extracted_data = '\n'.join(extracted_summary)
    
    prompt = VERIFICATION_PROMPT.format(
        extracted_data=extracted_data,
        document_text=document_sample[:10000]
    )
    
    messages = [
        {"role": "system", "content": "You are a financial analyst performing quality control. Respond only with valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    response = llm_generate(messages, model, tokenizer, max_tokens=3000)
    return parse_json_response(response) or {}


def apply_corrections(extractions: Dict, verification: Dict) -> Dict:
    """Apply corrections from verification pass."""
    if not verification or 'verified_fields' not in verification:
        return extractions
    
    corrections = verification['verified_fields']
    
    for group_name, group_data in extractions.items():
        if 'fields' not in group_data:
            continue
        
        for field_name, field_data in group_data['fields'].items():
            if field_name in corrections:
                correction = corrections[field_name]
                if not correction.get('is_correct', True):
                    verified_value = correction.get('verified_value')
                    if verified_value:
                        if isinstance(field_data, dict):
                            field_data['value'] = verified_value
                            field_data['confidence'] = 'MEDIUM'
                            field_data['evidence'] += f" [CORRECTED: {correction.get('correction_reason', '')}]"
    
    return extractions


# =====================================================================
# CONSOLIDATION
# =====================================================================

def consolidate_facility_data(extractions: Dict, facility_info: Dict) -> FacilityData:
    """Consolidate all extracted data for a single facility."""
    fields = {}
    
    for group_name, group_data in extractions.items():
        if 'fields' not in group_data:
            continue
        
        for field_name, field_data in group_data['fields'].items():
            if isinstance(field_data, dict):
                fields[field_name] = ExtractedValue(
                    value=field_data.get('value', 'NOT_FOUND'),
                    evidence=field_data.get('evidence', ''),
                    confidence=field_data.get('confidence', 'LOW')
                )
            else:
                fields[field_name] = ExtractedValue(
                    value=str(field_data),
                    evidence='',
                    confidence='LOW'
                )
    
    return FacilityData(
        facility_name=facility_info.get('name', 'Unknown'),
        facility_type=facility_info.get('type', 'Unknown'),
        fields=fields,
        raw_extractions=extractions
    )


def calculate_confidence_score(facility_data: FacilityData) -> float:
    """Calculate overall confidence score for extraction."""
    confidence_weights = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3}
    skip_values = {'NOT_FOUND', 'N/A', 'EXTRACTION_ERROR', '', 'POSSIBLY_PRESENT'}

    scores = []
    for field_name, field_value in facility_data.fields.items():
        if field_value.value not in skip_values:
            scores.append(confidence_weights.get(field_value.confidence, 0.3))

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


# =====================================================================
# MAIN PROCESSING
# =====================================================================

def process_document(company: str, manifest_entry: Dict,
                    model, tokenizer,
                    retriever_type: str = "bm25",
                    embedding_model=None,
                    embedding_tokenizer=None) -> Optional[DocumentExtraction]:
    """Process a single document through the full extraction pipeline."""

    # Load chunks
    chunks_file = os.path.join(CHUNKS_DIR, manifest_entry['chunks_file'])
    if not os.path.exists(chunks_file):
        logger.error(f"Chunks file not found: {chunks_file}")
        return None

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    if not chunks:
        logger.warning(f"No chunks in file: {chunks_file}")
        return None

    # Load full text for facility detection
    text_file = os.path.join(PREPROCESSED_DATA_DIR, manifest_entry['text_file'])
    with open(text_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Create retriever
    retriever = create_retriever_from_chunks(
        chunks, retriever_type=retriever_type,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer
    )
    
    # Step 1: Detect facilities
    logger.info("  Step 1: Detecting facilities...")
    facilities = detect_facilities(full_text, model, tokenizer)
    
    extracted_facilities = []
    
    # Step 2: Extract for each facility
    for i, facility_info in enumerate(facilities):
        facility_context = f"{facility_info.get('name', 'Facility')} ({facility_info.get('type', 'Unknown type')})"
        logger.info(f"  Step 2.{i+1}: Extracting for {facility_context}")
        
        # Multi-pass extraction
        extractions = extract_all_field_groups(retriever, facility_context, model, tokenizer)
        
        # Step 3: Verification (sample-based for speed)
        logger.info(f"  Step 3.{i+1}: Verifying extraction...")
        verification = verify_extraction(extractions, full_text, model, tokenizer)
        extractions = apply_corrections(extractions, verification)
        
        # Consolidate
        facility_data = consolidate_facility_data(extractions, facility_info)
        extracted_facilities.append(facility_data)
    
    # Calculate overall confidence
    avg_confidence = sum(calculate_confidence_score(f) for f in extracted_facilities) / len(extracted_facilities)
    
    return DocumentExtraction(
        company=company,
        source_file=manifest_entry['original_file'],
        facilities=extracted_facilities,
        document_summary=f"Extracted {len(extracted_facilities)} facilities",
        extraction_timestamp=datetime.now().isoformat(),
        confidence_score=avg_confidence
    )


def extraction_to_rows(extraction: DocumentExtraction) -> List[Dict]:
    """Convert extraction to flat rows for CSV."""
    rows = []
    
    for facility in extraction.facilities:
        row = {
            "Client Folder": extraction.company,
            "Source Files": extraction.source_file,
            "Comments": f"Facility: {facility.facility_name} | Type: {facility.facility_type}",
            "Confidence Score": f"{extraction.confidence_score:.2%}"
        }
        
        # Add all extracted fields (with normalization)
        for field_name in EXTRACTABLE_FIELDS:
            if field_name in facility.fields:
                field_data = facility.fields[field_name]
                # POSSIBLY_PRESENT means we couldn't find it despite trying — treat as NOT_FOUND in output
                if field_data.value == 'POSSIBLY_PRESENT':
                    row[field_name] = "NOT_FOUND"
                else:
                    row[field_name] = normalize_field_value(field_name, field_data.value)
            else:
                row[field_name] = "NOT_EXTRACTED"
        
        rows.append(row)
    
    return rows


def save_detailed_extraction(extraction: DocumentExtraction, output_dir: str):
    """Save detailed extraction with evidence for review."""
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', 
                       f"{extraction.company}_{extraction.source_file}")
    output_file = os.path.join(output_dir, f"{safe_name}.json")
    
    # Convert to serializable format
    output_data = {
        "company": extraction.company,
        "source_file": extraction.source_file,
        "timestamp": extraction.extraction_timestamp,
        "confidence_score": extraction.confidence_score,
        "facilities": []
    }
    
    for facility in extraction.facilities:
        fac_data = {
            "facility_name": facility.facility_name,
            "facility_type": facility.facility_type,
            "fields": {}
        }
        
        for field_name, field_value in facility.fields.items():
            fac_data["fields"][field_name] = {
                "value": field_value.value,
                "evidence": field_value.evidence,
                "confidence": field_value.confidence
            }
        
        output_data["facilities"].append(fac_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def run_extraction(company_filter: str = None, model=None, tokenizer=None):
    """
    Main extraction loop - COMPANY-LEVEL processing.
    
    Key change: Instead of processing files individually, we:
    1. Get unique companies
    2. Consolidate ALL files per company
    3. Extract from the combined document
    
    This ensures information scattered across files is captured.
    """
    
    # Load manifest
    manifest_path = os.path.join(PREPROCESSED_DATA_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        return
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Get unique companies
    if company_filter:
        companies = [company_filter]
        manifest = [m for m in manifest if m['company'] == company_filter]
    else:
        companies = get_all_companies(manifest)
    
    logger.info(f"Processing {len(companies)} companies ({len(manifest)} total files)")
    log_gpu_memory("Before extraction loop: ")
    
    # Create output directory
    os.makedirs(EXTRACTION_DIR, exist_ok=True)
    
    # Start memory monitor
    memory_monitor = GPUMemoryMonitor(interval_sec=30.0, log_threshold_pct=85.0)
    memory_monitor.start()
    
    all_rows = []
    
    try:
        for i, company in enumerate(companies):
            company_files = [m for m in manifest if m['company'] == company]
            
            logger.info(f"\n{'='*70}")
            logger.info(f"COMPANY {i+1}/{len(companies)}: {company}")
            logger.info(f"Files to consolidate: {len(company_files)}")
            logger.info(f"{'='*70}")
            
            try:
                with gpu_memory_context(f"company {company}"):
                    # Use consolidated company processing
                    extraction = process_company_consolidated(
                        company, manifest, model, tokenizer
                    )
                
                if extraction:
                    # Save detailed JSON
                    save_detailed_extraction(extraction, EXTRACTION_DIR)
                    
                    # Convert to rows
                    rows = extraction_to_rows(extraction)
                    all_rows.extend(rows)
                    
                    logger.info(f"  ✓ Extracted {len(extraction.facilities)} facilities from "
                               f"{len(company_files)} files, confidence: {extraction.confidence_score:.2%}")
                else:
                    logger.warning(f"  ✗ No data extracted for {company}")
            
            except torch.cuda.OutOfMemoryError:
                logger.error(f"  OOM Error - attempting recovery...")
                aggressive_memory_cleanup()
                continue
                
            except Exception as e:
                logger.error(f"  Failed to process {company}: {e}", exc_info=True)
            
            # Cleanup after each company
            aggressive_memory_cleanup()
            log_gpu_memory(f"After company {i+1}: ")
    
    finally:
        # Stop monitor and log peak usage
        memory_monitor.stop()
        logger.info(f"Peak GPU memory usage: {memory_monitor.peak_memory_gb:.2f}GB")
    
    # Save final CSV
    if all_rows:
        df = pd.DataFrame(all_rows)
        
        # Reorder columns
        ordered_cols = [c for c in ALL_FIELDS if c in df.columns]
        extra_cols = [c for c in df.columns if c not in ordered_cols]
        df = df[ordered_cols + extra_cols]
        
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        logger.info(f"\n{'='*70}")
        logger.info(f"SUCCESS: Saved {len(df)} rows to {OUTPUT_CSV}")
        logger.info(f"Detailed extractions saved to {EXTRACTION_DIR}/")
        logger.info(f"{'='*70}")
    else:
        logger.warning("No data extracted")


# =====================================================================
# CONSOLIDATED COMPANY PROCESSING
# =====================================================================

def process_company_consolidated(
    company: str,
    manifest: List[Dict],
    model,
    tokenizer,
    retriever_type: str = "bm25",
    embedding_model=None,
    embedding_tokenizer=None
) -> Optional[DocumentExtraction]:
    """
    Process ALL files for a company as a single consolidated document.

    This ensures:
    1. Information from all files is combined
    2. ALL text is processed by the LLM
    3. Related information across files is connected
    """

    # Step 1: Consolidate all files for this company
    logger.info(f"  Step 1: Consolidating all files...")

    consolidated = consolidate_company_documents(
        company, manifest, PREPROCESSED_DATA_DIR, tokenizer
    )

    if not consolidated or not consolidated.chunks:
        logger.warning(f"  No content found for {company}")
        return None

    logger.info(f"  Consolidated: {consolidated.total_tokens:,} tokens, "
               f"{len(consolidated.chunks)} chunks from {len(consolidated.source_files)} files")

    # Save consolidated document for debugging
    save_consolidated_document(consolidated, EXTRACTION_DIR)

    # Step 2: Create retriever from consolidated chunks
    retriever = create_retriever_from_chunks(
        consolidated.chunks, retriever_type=retriever_type,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer
    )
    
    # Step 3: Detect facilities from the combined document
    logger.info(f"  Step 2: Detecting facilities...")
    
    # Use beginning of document which typically has facility overview
    sample_text = consolidated.full_text[:20000]
    facilities = detect_facilities(sample_text, model, tokenizer, retriever=retriever)
    
    logger.info(f"  Found {len(facilities)} facilities")
    
    # Step 4: Extract for each facility using ALL chunks
    extracted_facilities = []
    
    for i, facility_info in enumerate(facilities):
        facility_context = f"{facility_info.get('name', 'Facility')} ({facility_info.get('type', 'Unknown')})"
        logger.info(f"  Step 3.{i+1}: Extracting for {facility_context}")
        
        # Multi-pass extraction covering ALL chunks
        all_extractions = extract_with_full_coverage(
            consolidated, retriever, facility_context, model, tokenizer
        )
        
        # Verification pass
        logger.info(f"  Step 4.{i+1}: Verifying extraction...")
        verification = verify_extraction(all_extractions, sample_text, model, tokenizer)
        all_extractions = apply_corrections(all_extractions, verification)
        
        # Consolidate facility data
        facility_data = consolidate_facility_data(all_extractions, facility_info)
        extracted_facilities.append(facility_data)
    
    # Calculate confidence
    avg_confidence = sum(calculate_confidence_score(f) for f in extracted_facilities) / len(extracted_facilities) if extracted_facilities else 0
    
    return DocumentExtraction(
        company=company,
        source_file=", ".join(consolidated.source_files),
        facilities=extracted_facilities,
        document_summary=f"Extracted {len(extracted_facilities)} facilities from {len(consolidated.source_files)} files",
        extraction_timestamp=datetime.now().isoformat(),
        confidence_score=avg_confidence
    )


def extract_with_full_coverage(
    consolidated: ConsolidatedDocument,
    retriever: BM25Retriever,
    facility_context: str,
    model,
    tokenizer
) -> Dict[str, Dict]:
    """
    Extract field groups ensuring ALL document content is analyzed.
    
    Strategy:
    1. For each field group, use BM25 to find most relevant chunks
    2. ALSO process document in sections to catch missed content
    3. Merge results keeping highest confidence values
    """
    all_extractions = {}
    
    for group_name, group_config in FIELD_GROUPS.items():
        logger.info(f"    Extracting: {group_config['name']}")
        
        # Get field names
        field_names = [name for name, _ in group_config['fields']]
        
        # Strategy 1: BM25 retrieval for most relevant chunks
        relevant_chunks = retrieve_for_field_group(
            retriever,
            group_config['keywords'],
            field_names,
            top_k=MAX_CHUNKS_PER_FIELD_GROUP
        )
        
        extraction_results = []
        
        # First extraction from BM25-retrieved chunks
        if relevant_chunks:
            result = extract_field_group(
                group_name, group_config, relevant_chunks, 
                facility_context, model, tokenizer
            )
            extraction_results.append(result)
        
        # Check if BM25 pass already found all fields with HIGH confidence
        all_found = False
        if extraction_results:
            found_count = 0
            for field_name, _ in group_config['fields']:
                fd = extraction_results[0].get('fields', {}).get(field_name)
                if fd and isinstance(fd, dict):
                    val = fd.get('value', 'NOT_FOUND')
                    conf = fd.get('confidence', 'LOW')
                    if val not in ('NOT_FOUND', 'N/A', '', None, 'POSSIBLY_PRESENT') and conf == 'HIGH':
                        found_count += 1
            if found_count == len(group_config['fields']):
                all_found = True
                logger.info(f"      All {found_count} fields found with HIGH confidence, skipping section passes")

        # Strategy 2: Process document in sections for full coverage (skip if all found)
        if not all_found:
            section_size = max(4, len(consolidated.chunks) // 3)  # ~3-4 sections

            for section_start in range(0, len(consolidated.chunks), section_size):
                section_end = min(section_start + section_size, len(consolidated.chunks))
                section_chunks = [c['text'] for c in consolidated.chunks[section_start:section_end]]

                if section_chunks:
                    # Check if this section content is already covered
                    section_preview = ' '.join(section_chunks)[:300]
                    already_covered = any(
                        section_preview[:100] in rc for rc in relevant_chunks
                    ) if relevant_chunks else False

                    if not already_covered:
                        result = extract_field_group(
                            group_name, group_config, section_chunks,
                            facility_context, model, tokenizer
                        )
                        extraction_results.append(result)
        
        # Merge results from all passes
        if extraction_results:
            merged = merge_field_extractions(extraction_results, group_config['fields'])
            all_extractions[group_name] = merged
        else:
            all_extractions[group_name] = {"fields": {}}
        
        # Memory cleanup between groups
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_extractions


def merge_field_extractions(results: List[Dict], fields: List[Tuple]) -> Dict:
    """Merge extraction results from multiple passes.

    POSSIBLY_PRESENT values are treated as "not yet found" during merging
    but preserved if no better value exists — this signals deep extraction
    to try harder for these fields.
    """
    merged = {"fields": {}, "notes": "Merged from multiple extraction passes"}
    skip_values = {'NOT_FOUND', 'N/A', '', None, 'POSSIBLY_PRESENT'}

    for field_name, _ in fields:
        best_value = None
        best_confidence = 'LOW'
        best_evidence = ''
        has_possibly_present = False
        conf_rank = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}

        for result in results:
            if 'fields' not in result:
                continue

            field_data = result['fields'].get(field_name)
            if not field_data or not isinstance(field_data, dict):
                continue

            value = field_data.get('value', 'NOT_FOUND')
            confidence = field_data.get('confidence', 'LOW')
            evidence = field_data.get('evidence', '')

            if value == 'POSSIBLY_PRESENT':
                has_possibly_present = True
                continue

            if value in skip_values:
                continue

            # Keep highest confidence value
            if conf_rank.get(confidence, 0) > conf_rank.get(best_confidence, 0):
                best_value = value
                best_confidence = confidence
                best_evidence = evidence
            elif conf_rank.get(confidence, 0) == conf_rank.get(best_confidence, 0):
                # Same confidence, prefer longer evidence
                if len(evidence) > len(best_evidence):
                    best_value = value
                    best_confidence = confidence
                    best_evidence = evidence

        # If no concrete value found but LLM hinted the field exists,
        # mark as POSSIBLY_PRESENT so deep extraction will target it
        if best_value is None and has_possibly_present:
            merged["fields"][field_name] = {
                "value": "POSSIBLY_PRESENT",
                "confidence": "LOW",
                "evidence": ""
            }
        else:
            merged["fields"][field_name] = {
                "value": best_value or "NOT_FOUND",
                "confidence": best_confidence,
                "evidence": best_evidence
            }

    return merged


def main():
    parser = argparse.ArgumentParser(description="Multi-Pass Evidence-Based Extraction v2")
    parser.add_argument("--company", help="Process specific company only")
    parser.add_argument("--flash_attention", action="store_true", help="Use Flash Attention 2")
    args = parser.parse_args()
    
    # Initialize model
    model, tokenizer = initialize_model(MODEL_PATH, args.flash_attention)
    
    if model is None:
        logger.error("Failed to initialize model")
        return
    
    # Run extraction
    run_extraction(args.company, model, tokenizer)


if __name__ == "__main__":
    main()
