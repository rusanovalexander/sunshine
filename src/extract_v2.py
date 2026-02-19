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
# DEBUG TRACE — saves every LLM call's input/output for inspection
# =====================================================================
import time as _time

DEBUG_TRACE_DIR = os.path.join(EXTRACTION_DIR, "debug_traces")
_trace_counter = 0


def _save_debug_trace(step_name: str, messages: list, response: str,
                      input_tokens: int, elapsed_sec: float,
                      parsed_result=None, company: str = "") -> str:
    """Save a single LLM call's full prompt + response for debugging.

    Returns the filepath of the saved trace (used by _update_trace_parsed_result).
    """
    global _trace_counter
    _trace_counter += 1

    # Create per-company subfolder
    safe_company = re.sub(r'[^a-zA-Z0-9_-]', '_', company) if company else "unknown"
    trace_dir = os.path.join(DEBUG_TRACE_DIR, safe_company)
    os.makedirs(trace_dir, exist_ok=True)

    trace = {
        "call_number": _trace_counter,
        "step": step_name,
        "company": company,
        "timestamp": datetime.now().isoformat(),
        "input_tokens": input_tokens,
        "elapsed_seconds": round(elapsed_sec, 2),
        "tokens_per_second": round(len(response.split()) / max(elapsed_sec, 0.01), 1),
        "messages": [
            {"role": m["role"], "content": m["content"]} for m in messages
        ],
        "raw_response": response,
        "parsed_ok": parsed_result is not None,
        "parsed_result": parsed_result,
    }

    filename = f"{_trace_counter:04d}_{step_name}.json"
    filepath = os.path.join(trace_dir, filename)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning(f"Failed to save debug trace: {e}")
    return filepath


def _update_trace_parsed_result(trace_filepath: str, parsed_result):
    """Patch an existing debug trace file with the parsed JSON result.

    Called after parse_json_response() so the trace shows whether
    parsing succeeded and what the structured result looks like.
    """
    try:
        with open(trace_filepath, 'r', encoding='utf-8') as f:
            trace = json.load(f)
        trace["parsed_ok"] = parsed_result is not None
        trace["parsed_result"] = parsed_result
        with open(trace_filepath, 'w', encoding='utf-8') as f:
            json.dump(trace, f, indent=2, ensure_ascii=False, default=str)
    except Exception:
        pass  # Non-critical — don't break extraction if trace update fails


# Current company context for debug traces (set by processing functions)
_current_company = ""
_current_language = "en"  # Detected document language (ISO 639-1 code)


def set_debug_company(company: str):
    """Set current company for debug trace file organization."""
    global _current_company
    _current_company = company


def detect_document_language(text: str) -> str:
    """Detect language of the document using langdetect.

    Returns ISO 639-1 code (e.g. 'en', 'es', 'fr', 'de', 'nl', 'pt').
    Falls back to 'en' if detection fails.
    """
    global _current_language
    try:
        from langdetect import detect
        # Use first ~5000 chars for reliable detection, skip headers/markers
        sample = re.sub(r'={3,}|\[PAGE \d+\]|\[SOURCE FILE:.*?\]', '', text[:8000])
        lang = detect(sample.strip())
        _current_language = lang
        logger.info(f"  Document language detected: {lang}")
        return lang
    except Exception as e:
        logger.warning(f"  Language detection failed ({e}), defaulting to English")
        _current_language = "en"
        return "en"


def _get_language_prompt_hint() -> str:
    """Return a prompt hint for non-English documents.

    Tells the LLM the document language so it can read non-English text
    correctly while still outputting field values and JSON keys in English.
    """
    if _current_language == "en":
        return ""

    LANG_NAMES = {
        "es": "Spanish", "fr": "French", "de": "German", "nl": "Dutch",
        "pt": "Portuguese", "it": "Italian", "pl": "Polish", "ro": "Romanian",
        "cs": "Czech", "da": "Danish", "sv": "Swedish", "nb": "Norwegian",
        "fi": "Finnish", "hu": "Hungarian", "el": "Greek", "tr": "Turkish",
        "ru": "Russian", "uk": "Ukrainian", "ja": "Japanese", "zh-cn": "Chinese",
        "ko": "Korean", "ar": "Arabic",
    }
    lang_name = LANG_NAMES.get(_current_language, _current_language.upper())

    return (
        f"\n\nLANGUAGE NOTE: This document is written in {lang_name}. "
        f"Read and understand the {lang_name} text, but output ALL field names "
        f"and JSON keys in English exactly as specified above. "
        f"Extract values as they appear in the document (keep original language "
        f"for names and terms, but convert dates/numbers to the requested format)."
    )


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
    """Load Qwen3-14B with 4-bit quantization.

    Uses the original proven config that works on A100 MIG 4g.20gb.
    Falls through to load_llm_optimized only for pre-quantized models.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading model: {model_path}")
    log_gpu_memory("Before model load: ")

    try:
        # Check if model is pre-quantized (FP8/GPTQ/AWQ/saved BnB)
        from .gpu_optimizer import _detect_quantization
        quant_type = _detect_quantization(model_path)

        if quant_type != "none":
            # Use optimized loader for pre-quantized models
            logger.info(f"  Pre-quantized model detected ({quant_type}), using optimized loader")
            model, tokenizer = load_llm_optimized(
                model_path,
                use_flash_attention=use_flash_attention,
            )
        else:
            # Original working config — proven on A100 MIG 4g.20gb
            logger.info(f"  Using original 4-bit config (sdpa attn, no max_memory cap)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, padding_side='left'
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        logger.info("Model loaded successfully")
        log_gpu_memory("After model load: ")
        return model, tokenizer
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        return None, None


def llm_generate(prompt_messages: List[Dict], model, tokenizer,
                 max_tokens: int = MAX_NEW_TOKENS,
                 step_name: str = "llm_call") -> Tuple[str, str]:
    """Generate LLM response with memory-efficient settings.

    Args:
        step_name: Label for debug trace (e.g. "facility_detection", "extract_pricing").

    Returns:
        (response_text, trace_filepath) — caller should update trace with parsed result
        via _update_trace_parsed_result(trace_filepath, parsed_result).
    """
    t0 = _time.time()
    try:
        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        # Log input size for debugging memory issues
        logger.info(f"  LLM input: {input_len} tokens, max_new_tokens={max_tokens}")

        # Dynamically cap max_new_tokens based on available memory
        stats = get_gpu_memory_stats()
        if stats:
            if stats.free_gb < 2.0:
                logger.warning(f"Very low GPU memory ({stats.free_gb:.2f}GB free), reducing to 256 tokens")
                max_tokens = min(max_tokens, 256)
                aggressive_memory_cleanup()
            elif stats.free_gb < 4.0:
                logger.warning(f"Low GPU memory ({stats.free_gb:.2f}GB free), reducing to 512 tokens")
                max_tokens = min(max_tokens, 512)
            elif stats.free_gb < 6.0:
                max_tokens = min(max_tokens, 1024)

        # Truncate input if too long for available memory
        # Rule of thumb: need ~0.5GB per 1K input tokens for KV cache on 14B model
        max_safe_input = 8000  # tokens — safe for ~11GB free VRAM
        if input_len > max_safe_input:
            logger.warning(f"Input too long ({input_len} tokens), truncating to {max_safe_input}")
            inputs = tokenizer(text, return_tensors="pt", max_length=max_safe_input, truncation=True).to(model.device)
            input_len = inputs.input_ids.shape[1]

        gen_kwargs = get_memory_efficient_generation_kwargs()
        gen_kwargs["max_new_tokens"] = max_tokens
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        response = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        ).strip()

        # Cleanup
        del inputs, outputs
        torch.cuda.empty_cache()

        # Save debug trace (parsed_result filled in later by caller)
        elapsed = _time.time() - t0
        _last_trace_path = _save_debug_trace(
            step_name=step_name, messages=prompt_messages,
            response=response, input_tokens=input_len,
            elapsed_sec=elapsed, company=_current_company
        )

        return response, _last_trace_path
    except Exception as e:
        # Catch ALL errors including NVML assertion failures (not just torch.cuda.OutOfMemoryError)
        error_str = str(e)
        is_oom = "NVML_SUCCESS" in error_str or "out of memory" in error_str.lower() or isinstance(e, torch.cuda.OutOfMemoryError)

        if is_oom:
            logger.error(f"GPU OOM during generation ({input_len} input tokens), attempting recovery...")
        else:
            logger.error(f"LLM generation error: {e}")

        aggressive_memory_cleanup()

        if is_oom:
            # Retry with drastically reduced tokens
            try:
                inputs = tokenizer(text, return_tensors="pt", max_length=4000, truncation=True).to(model.device)
                retry_len = inputs.input_ids.shape[1]
                logger.info(f"  Retry with {retry_len} input tokens, 256 max_new_tokens")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=256, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                response = tokenizer.decode(
                    outputs[0][retry_len:], skip_special_tokens=True
                ).strip()
                del inputs, outputs
                torch.cuda.empty_cache()
                return response, ""
            except Exception as e2:
                logger.error(f"Recovery failed: {e2}")
                aggressive_memory_cleanup()
                return "", ""

        return "", ""


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


def validate_extraction_schema(result: Dict, expected_fields: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that an extraction result has the expected JSON schema.

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    if not isinstance(result, dict):
        return False, ["Result is not a dict"]

    if "fields" not in result:
        issues.append("Missing 'fields' key")
        return False, issues

    fields = result["fields"]
    if not isinstance(fields, dict):
        issues.append("'fields' is not a dict")
        return False, issues

    # Check each expected field is present with valid structure
    for field_name in expected_fields:
        if field_name not in fields:
            issues.append(f"Missing field: {field_name}")
            continue

        field_data = fields[field_name]
        if not isinstance(field_data, dict):
            issues.append(f"Field '{field_name}' is not a dict (got {type(field_data).__name__})")
            continue

        if "value" not in field_data:
            issues.append(f"Field '{field_name}' missing 'value' key")

        if "confidence" not in field_data:
            issues.append(f"Field '{field_name}' missing 'confidence' key")

    is_valid = len(issues) == 0
    return is_valid, issues


# =====================================================================
# MULTILINGUAL RETRIEVAL QUERIES
# =====================================================================

# BM25 queries for facility detection in non-English documents.
# Each language maps to queries covering: facility enumeration, commitments,
# definitions, maturity, and pricing — mirroring the English queries.
_MULTILINGUAL_FACILITY_QUERIES = {
    "es": [  # Spanish — covers both LMA-style and notarial deed vocabulary
        "Las Facilidades poner a disposición préstamo crédito monto total importe",
        "Compromisos Totales compromiso agregado facilidad",
        "facilidad tramo compromiso importe límite crédito",
        "Fecha de Vencimiento plazo años desde amortización reembolso",
        "Margen porcentaje por ciento anual facilidad préstamo tipo de interés",
        # Notarial deed / escritura pública terms
        "El Crédito disponibilidad importe principal entidades acreedoras",
        "contrato de crédito sindicado préstamo participantes agente",
        "Reembolso Amortización vencimiento cuotas calendario pagos",
    ],
    "fr": [  # French
        "Les Facilités mettre à disposition prêt crédit montant total",
        "Engagements Totaux engagement agrégé facilité",
        "facilité tranche engagement montant limite crédit",
        "Date d'Échéance années à compter de",
        "Marge pourcentage par an facilité prêt",
    ],
    "de": [  # German
        "Die Fazilitäten zur Verfügung stellen Darlehen Kredit Gesamtbetrag",
        "Gesamtzusagen Zusage aggregiert Fazilität",
        "Fazilität Tranche Zusage Betrag Limit Kredit",
        "Endfälligkeitsdatum Jahre ab Lieferung",
        "Marge Prozent pro Jahr Fazilität Darlehen",
    ],
    "nl": [  # Dutch
        "De Faciliteiten beschikbaar stellen lening krediet totaal bedrag",
        "Totale Verplichtingen verplichting geaggregeerd faciliteit",
        "faciliteit tranche verplichting bedrag limiet krediet",
        "Einddatum looptijd jaren vanaf levering",
        "Marge percentage per jaar faciliteit lening",
    ],
    "pt": [  # Portuguese
        "As Facilidades disponibilizar empréstimo crédito montante total",
        "Compromissos Totais compromisso agregado facilidade",
        "facilidade tranche compromisso montante limite crédito",
        "Data de Vencimento prazo anos a partir de",
        "Margem percentagem por cento por ano facilidade empréstimo",
    ],
    "it": [  # Italian
        "Le Facilitazioni mettere a disposizione prestito credito importo totale",
        "Impegni Totali impegno aggregato facilitazione",
        "facilitazione tranche impegno importo limite credito",
        "Data di Scadenza anni dalla consegna",
        "Margine percentuale per cento annuo facilitazione prestito",
    ],
    "pl": [  # Polish
        "Udostępnić kredyt pożyczka kwota łączna suma",
        "Zobowiązania łączne zobowiązanie kredyt",
        "kredyt transza zobowiązanie kwota limit",
        "Data zapadalności termin lata od",
        "Marża procent rocznie kredyt pożyczka",
    ],
}


def _get_multilingual_facility_queries(lang_code: str) -> List[str]:
    """Return facility detection BM25 queries for the given language."""
    return _MULTILINGUAL_FACILITY_QUERIES.get(lang_code, [])


# =====================================================================
# FACILITY DETECTION
# =====================================================================

def detect_facilities(document_text: str, model, tokenizer,
                     retriever: BM25Retriever = None) -> List[Dict]:
    """
    First pass: Detect all distinct facilities/tranches in the document.
    This helps us know if we need per-facility extraction.

    Uses targeted retrieval with multiple queries to gather all
    facility-related sections, not just the document beginning.

    Strategy: collect ALL candidate chunks (retrieved + document edges),
    deduplicate, rank by relevance score, and greedily fill the 5K token
    budget with the best chunks.  No hardcoded ordering — works for any
    document language or structure.
    """
    # Collect scored candidate chunks: (score, text)
    # Higher score = more likely to contain facility definitions
    candidates: Dict[str, float] = {}  # prefix -> best score
    candidate_texts: Dict[str, str] = {}  # prefix -> full text

    def _add_candidate(text: str, score: float):
        """Add candidate chunk, keeping highest score per unique prefix."""
        prefix = text[:200].strip()
        if prefix not in candidates or score > candidates[prefix]:
            candidates[prefix] = score
            candidate_texts[prefix] = text

    # 1. Targeted retrieval with multiple specific queries
    if retriever:
        targeted_queries = [
            # English queries
            "The Facilities make available term loan revolving credit facility aggregate amount",
            "Total Commitments Total Facility Commitments aggregate being",
            "facility tranche commitment amount schedule limit",
            "Final Maturity Date years from Escrow Delivery",
            "Margin per cent per annum Facility Loan",
        ]

        # Add multilingual queries for non-English documents
        if _current_language != "en":
            multilingual_queries = _get_multilingual_facility_queries(_current_language)
            targeted_queries.extend(multilingual_queries)
        for query in targeted_queries:
            results = retriever.search(query, top_k=3)
            for _, score, chunk_text in results:
                _add_candidate(chunk_text, score)

    # 2. Beginning of document (cover page, TOC, initial definitions)
    # Score 0.1 — low baseline so retriever hits beat it when relevant,
    # but it still gets included when budget allows
    _add_candidate(document_text[:8000], 0.1)

    # 3. End of document (schedules with lender commitments)
    if len(document_text) > 20000:
        _add_candidate(document_text[-5000:], 0.05)

    # Sort candidates by score descending and greedily fill token budget
    max_detect_tokens = 5000
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    selected_parts = []
    used_tokens = 0
    for prefix, score in sorted_candidates:
        text = candidate_texts[prefix]
        part_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        if used_tokens + part_tokens <= max_detect_tokens:
            selected_parts.append(text)
            used_tokens += part_tokens
        elif used_tokens < max_detect_tokens:
            # Partially fit the last chunk
            remaining = max_detect_tokens - used_tokens
            part_tok_ids = tokenizer.encode(text, add_special_tokens=False)[:remaining]
            selected_parts.append(tokenizer.decode(part_tok_ids, skip_special_tokens=True))
            used_tokens += remaining
            break

    combined = "\n\n---\n\n".join(selected_parts)

    logger.info(f"      Facility detection: {len(candidates)} unique chunks scored, "
                f"{len(selected_parts)} selected → {used_tokens} tokens")

    prompt = FACILITY_DETECTION_PROMPT.format(document_text=combined)
    prompt += _get_language_prompt_hint()

    messages = [
        {"role": "system", "content": "You are a financial analyst. Respond only with valid JSON."},
        {"role": "user", "content": prompt}
    ]

    response, trace_path = llm_generate(messages, model, tokenizer, max_tokens=2048,
                                         step_name="facility_detection")
    result = parse_json_response(response)
    _update_trace_parsed_result(trace_path, result)

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

    # Pre-truncate document text to ensure the FULL prompt (with JSON format
    # instructions at the end) fits within the safe input limit.
    # Budget: ~1500 tokens for system prompt + field descriptions + format instructions
    # Remaining: 8000 - 1500 = 6500 tokens max for document text
    max_doc_tokens = 6500
    doc_tokens = tokenizer.encode(document_text, add_special_tokens=False)
    if len(doc_tokens) > max_doc_tokens:
        logger.info(f"      Trimming document text from {len(doc_tokens)} to {max_doc_tokens} tokens")
        document_text = tokenizer.decode(doc_tokens[:max_doc_tokens], skip_special_tokens=True)

    # Build prompt
    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        fields_description=fields_description,
        document_text=document_text
    )

    # Enhance with per-group few-shot example
    if FEW_SHOT_AVAILABLE:
        user_prompt = create_enhanced_extraction_prompt(user_prompt, group_name=group_name)

    # Add language hint for non-English documents
    user_prompt += _get_language_prompt_hint()

    # Use group-specific system prompt if available, otherwise generic
    system_prompt = FIELD_GROUP_SYSTEM_PROMPTS.get(group_name, EXTRACTION_SYSTEM_PROMPT)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Use adaptive max_new_tokens per field group to save KV cache memory
    group_max_tokens = FIELD_GROUP_MAX_TOKENS.get(group_name, MAX_NEW_TOKENS)

    # Expected field names for schema validation
    expected_field_names = [name for name, _ in fields]

    # Attempt 1: normal generation
    response, trace_path = llm_generate(messages, model, tokenizer, max_tokens=group_max_tokens,
                                        step_name=f"extract_{group_name}")
    result = parse_json_response(response)
    _update_trace_parsed_result(trace_path, result)

    if result:
        is_valid, issues = validate_extraction_schema(result, expected_field_names)
        if is_valid:
            return result
        else:
            logger.warning(f"      Schema validation issues: {issues[:3]}...")  # Show first 3
            # Try to fix: if 'fields' key exists but some fields missing,
            # fill them with NOT_FOUND rather than retrying
            if "fields" in result and isinstance(result["fields"], dict):
                for field_name in expected_field_names:
                    if field_name not in result["fields"]:
                        result["fields"][field_name] = {
                            "value": "NOT_FOUND",
                            "evidence": "",
                            "confidence": "LOW"
                        }
                    elif not isinstance(result["fields"][field_name], dict):
                        # Field exists but wrong type — wrap it
                        raw_val = result["fields"][field_name]
                        result["fields"][field_name] = {
                            "value": str(raw_val),
                            "evidence": "",
                            "confidence": "LOW"
                        }
                return result

    # Attempt 2: retry with explicit JSON repair instruction
    logger.info(f"      Retrying extraction for {group_name} (JSON parse/validation failed)")
    retry_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt + "\n\nIMPORTANT: Your response MUST be valid JSON only. No text before or after the JSON object."}
    ]
    response2, trace_path2 = llm_generate(retry_messages, model, tokenizer, max_tokens=group_max_tokens,
                                           step_name=f"extract_{group_name}_retry")
    result2 = parse_json_response(response2)
    _update_trace_parsed_result(trace_path2, result2)

    if result2 and "fields" in result2:
        # Patch missing fields
        for field_name in expected_field_names:
            if field_name not in result2.get("fields", {}):
                result2.setdefault("fields", {})[field_name] = {
                    "value": "NOT_FOUND", "evidence": "", "confidence": "LOW"
                }
        return result2

    # All attempts failed — return error structure
    return {
        "fields": {name: {"value": "EXTRACTION_ERROR", "evidence": "", "confidence": "LOW"}
                  for name, _ in fields},
        "notes": "Failed to parse extraction response after retry"
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
                     model, tokenizer,
                     retriever: BM25Retriever = None) -> Dict:
    """
    Verification pass: Check extracted values against source.
    Uses retriever to pull relevant context for the extracted values,
    instead of just the document beginning.
    """
    # Collect all extracted values
    extracted_summary = []
    verification_keywords = []
    for group_name, group_data in extractions.items():
        if 'fields' not in group_data:
            continue
        for field_name, field_data in group_data['fields'].items():
            if isinstance(field_data, dict):
                value = field_data.get('value', 'N/A')
                if value not in ('NOT_FOUND', 'N/A', '', 'EXTRACTION_ERROR', 'POSSIBLY_PRESENT'):
                    extracted_summary.append(f"- {field_name}: {value}")
                    # Collect keywords from values for retrieval
                    verification_keywords.append(f"{field_name} {value}")

    extracted_data = '\n'.join(extracted_summary)

    # Build verification context using retriever instead of blind truncation
    if retriever and verification_keywords:
        context_parts = []
        seen = set()
        # Retrieve chunks relevant to the extracted values
        for kw in verification_keywords[:8]:  # Limit queries
            results = retriever.search(kw, top_k=2)
            for _, _, chunk_text in results:
                prefix = chunk_text[:200]
                if prefix not in seen:
                    seen.add(prefix)
                    context_parts.append(chunk_text)
        verification_context = '\n\n---\n\n'.join(context_parts[:6])
    else:
        verification_context = document_sample[:5000]

    # Token-truncate the context to fit within input budget
    max_verify_tokens = 4000
    ctx_tokens = tokenizer.encode(verification_context, add_special_tokens=False)
    if len(ctx_tokens) > max_verify_tokens:
        verification_context = tokenizer.decode(ctx_tokens[:max_verify_tokens], skip_special_tokens=True)

    prompt = VERIFICATION_PROMPT.format(
        extracted_data=extracted_data,
        document_text=verification_context
    )
    prompt += _get_language_prompt_hint()

    messages = [
        {"role": "system", "content": "You are a financial analyst performing quality control. Respond only with valid JSON."},
        {"role": "user", "content": prompt}
    ]

    response, trace_path = llm_generate(messages, model, tokenizer, max_tokens=1024,
                                         step_name="verification")
    result = parse_json_response(response)
    _update_trace_parsed_result(trace_path, result)
    return result or {}


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
    set_debug_company(company)

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

    # Detect document language for multilingual support
    detect_document_language(full_text)

    # Create retriever
    retriever = create_retriever_from_chunks(
        chunks, retriever_type=retriever_type,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer
    )

    # Step 1: Detect facilities (pass retriever for targeted section retrieval)
    logger.info("  Step 1: Detecting facilities...")
    facilities = detect_facilities(full_text, model, tokenizer, retriever=retriever)

    extracted_facilities = []

    # Step 2: Extract for each facility
    for i, facility_info in enumerate(facilities):
        facility_context = f"{facility_info.get('name', 'Facility')} ({facility_info.get('type', 'Unknown type')})"
        logger.info(f"  Step 2.{i+1}: Extracting for {facility_context}")
        
        # Multi-pass extraction
        extractions = extract_all_field_groups(retriever, facility_context, model, tokenizer)
        
        # Step 3: Verification (retriever-based for relevant context)
        logger.info(f"  Step 3.{i+1}: Verifying extraction...")
        verification = verify_extraction(extractions, full_text, model, tokenizer, retriever=retriever)
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
    set_debug_company(company)

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

    # Detect document language for multilingual support
    detect_document_language(consolidated.full_text)

    # Step 2: Create retriever from consolidated chunks
    retriever = create_retriever_from_chunks(
        consolidated.chunks, retriever_type=retriever_type,
        embedding_model=embedding_model,
        embedding_tokenizer=embedding_tokenizer
    )
    
    # Step 3: Detect facilities from the combined document
    logger.info(f"  Step 2: Detecting facilities...")

    # Pass full text — detect_facilities uses targeted BM25 retrieval
    # to find facility-related sections, then token-truncates internally
    facilities = detect_facilities(consolidated.full_text, model, tokenizer, retriever=retriever)
    
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
        
        # Verification pass (retriever-based for relevant context)
        logger.info(f"  Step 4.{i+1}: Verifying extraction...")
        verification = verify_extraction(all_extractions, consolidated.full_text, model, tokenizer, retriever=retriever)
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
        # Each section pass sends MAX_CHUNKS_PER_FIELD_GROUP chunks to stay within VRAM
        if not all_found:
            # Track which chunks were already covered by BM25 retrieval
            bm25_chunk_texts = set()
            if relevant_chunks:
                for rc in relevant_chunks:
                    # Use first 300 chars as fingerprint (enough to uniquely ID a chunk)
                    bm25_chunk_texts.add(rc[:300])

            chunk_limit = MAX_CHUNKS_PER_FIELD_GROUP

            for section_start in range(0, len(consolidated.chunks), chunk_limit):
                section_end = min(section_start + chunk_limit, len(consolidated.chunks))
                section_chunks = [c['text'] for c in consolidated.chunks[section_start:section_end]]

                if section_chunks:
                    # Check overlap: skip if ALL chunks in this section were in BM25 results
                    covered_count = sum(
                        1 for sc in section_chunks
                        if sc[:300] in bm25_chunk_texts
                    )
                    if covered_count == len(section_chunks):
                        continue  # Fully covered by BM25 pass

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
