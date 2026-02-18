"""
=================================================
Company Document Consolidation Module
=================================================
Combines all preprocessed files for a company into
a single unified document for comprehensive extraction.

Key features:
1. Merges all text files per company
2. Preserves source file references
3. Creates combined chunks with full coverage
4. Ensures NO information is lost
"""

import os
import json
import logging
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SourceReference:
    """Tracks which file content came from."""
    file_name: str
    start_char: int
    end_char: int
    start_chunk: int
    end_chunk: int


@dataclass 
class ConsolidatedDocument:
    """A company's combined document."""
    company: str
    full_text: str
    source_files: List[str]
    source_references: List[SourceReference]
    total_chars: int
    total_tokens: int
    chunks: List[Dict]


def consolidate_company_documents(
    company: str,
    manifest: List[Dict],
    preprocessed_dir: str,
    tokenizer
) -> ConsolidatedDocument:
    """
    Combine all preprocessed files for a company into one document.
    
    Args:
        company: Company name
        manifest: Full manifest list
        preprocessed_dir: Directory with preprocessed text files
        tokenizer: For token counting and chunking
    
    Returns:
        ConsolidatedDocument with all company text combined
    """
    # Filter manifest for this company
    company_files = [m for m in manifest if m['company'] == company]
    
    if not company_files:
        logger.warning(f"No files found for company: {company}")
        return None
    
    logger.info(f"Consolidating {len(company_files)} files for {company}")
    
    # Combine all text with source markers
    combined_parts = []
    source_references = []
    source_files = []
    current_pos = 0
    
    for entry in company_files:
        text_path = os.path.join(preprocessed_dir, entry['text_file'])
        
        if not os.path.exists(text_path):
            logger.warning(f"Text file not found: {text_path}")
            continue
        
        with open(text_path, 'r', encoding='utf-8') as f:
            file_text = f.read()
        
        if not file_text.strip():
            continue
        
        # Add source header
        source_header = f"\n{'='*80}\n[SOURCE FILE: {entry['original_file']}]\n{'='*80}\n"
        
        combined_parts.append(source_header)
        combined_parts.append(file_text)
        combined_parts.append("\n\n")
        
        # Track source reference
        start_pos = current_pos + len(source_header)
        end_pos = start_pos + len(file_text)
        
        source_references.append(SourceReference(
            file_name=entry['original_file'],
            start_char=start_pos,
            end_char=end_pos,
            start_chunk=-1,  # Will be set during chunking
            end_chunk=-1
        ))
        
        source_files.append(entry['original_file'])
        current_pos = end_pos + 2  # +2 for \n\n
    
    # Combine into full text
    full_text = ''.join(combined_parts)
    
    # Count tokens
    if tokenizer:
        total_tokens = len(tokenizer.encode(full_text, add_special_tokens=False))
    else:
        total_tokens = len(full_text) // 4  # Rough estimate
    
    logger.info(f"  Combined document: {len(full_text):,} chars, ~{total_tokens:,} tokens")
    logger.info(f"  Source files: {len(source_files)}")
    
    # Create chunks with overlap
    chunks = create_comprehensive_chunks(
        full_text, 
        tokenizer, 
        source_references,
        chunk_size=3000,  # Larger chunks for better context
        overlap=500
    )
    
    return ConsolidatedDocument(
        company=company,
        full_text=full_text,
        source_files=source_files,
        source_references=source_references,
        total_chars=len(full_text),
        total_tokens=total_tokens,
        chunks=chunks
    )


def create_comprehensive_chunks(
    text: str,
    tokenizer,
    source_refs: List[SourceReference],
    chunk_size: int = 3000,
    overlap: int = 500
) -> List[Dict]:
    """
    Create overlapping chunks that cover the ENTIRE document.
    
    CRITICAL: Every character must be in at least one chunk.
    Uses larger chunks with more overlap for better context.
    """
    if not tokenizer:
        # Fallback: character-based chunking
        return _chunk_by_chars(text, source_refs, chunk_size * 4, overlap * 4)
    
    chunks = []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)
    
    logger.info(f"  Creating chunks: {total_tokens} tokens, size={chunk_size}, overlap={overlap}")
    
    position = 0
    chunk_id = 0
    
    while position < total_tokens:
        # Get chunk end position
        end_pos = min(position + chunk_size, total_tokens)
        
        # Extract chunk tokens and decode
        chunk_tokens = tokens[position:end_pos]
        chunk_text = tokenizer.decode(chunk_tokens)
        
        # Calculate character positions for source tracking
        char_start = len(tokenizer.decode(tokens[:position])) if position > 0 else 0
        char_end = char_start + len(chunk_text)
        
        # Find which source files this chunk covers
        chunk_sources = []
        for ref in source_refs:
            if ref.start_char < char_end and ref.end_char > char_start:
                chunk_sources.append(ref.file_name)
                # Update chunk range in source reference
                if ref.start_chunk == -1:
                    ref.start_chunk = chunk_id
                ref.end_chunk = chunk_id
        
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "token_start": position,
            "token_end": end_pos,
            "char_start": char_start,
            "char_end": char_end,
            "source_files": list(set(chunk_sources)),
            "is_last": end_pos >= total_tokens
        })
        
        chunk_id += 1
        
        # Move position with overlap
        if end_pos >= total_tokens:
            break
        position = end_pos - overlap
    
    logger.info(f"  Created {len(chunks)} chunks covering entire document")
    
    # Verify complete coverage
    _verify_coverage(chunks, total_tokens, text)
    
    return chunks


def _chunk_by_chars(
    text: str, 
    source_refs: List[SourceReference],
    chunk_size: int,
    overlap: int
) -> List[Dict]:
    """Fallback character-based chunking."""
    chunks = []
    position = 0
    chunk_id = 0
    
    while position < len(text):
        end_pos = min(position + chunk_size, len(text))
        chunk_text = text[position:end_pos]
        
        # Find source files
        chunk_sources = []
        for ref in source_refs:
            if ref.start_char < end_pos and ref.end_char > position:
                chunk_sources.append(ref.file_name)
        
        chunks.append({
            "id": chunk_id,
            "text": chunk_text,
            "token_count": len(chunk_text) // 4,
            "char_start": position,
            "char_end": end_pos,
            "source_files": list(set(chunk_sources)),
            "is_last": end_pos >= len(text)
        })
        
        chunk_id += 1
        
        if end_pos >= len(text):
            break
        position = end_pos - overlap
    
    return chunks


def _verify_coverage(chunks: List[Dict], total_tokens: int, text: str):
    """Verify that chunks cover the entire document."""
    if not chunks:
        logger.error("No chunks created!")
        return
    
    # Check token coverage
    covered_tokens = set()
    for chunk in chunks:
        for t in range(chunk['token_start'], chunk['token_end']):
            covered_tokens.add(t)
    
    coverage_pct = len(covered_tokens) / total_tokens * 100
    
    if coverage_pct < 99.9:
        logger.warning(f"  INCOMPLETE COVERAGE: {coverage_pct:.2f}% tokens covered")
        # Find gaps
        all_tokens = set(range(total_tokens))
        missing = all_tokens - covered_tokens
        if missing:
            logger.warning(f"  Missing token positions: {sorted(list(missing))[:10]}...")
    else:
        logger.info(f"  ✓ Complete coverage verified: {coverage_pct:.2f}%")


def save_consolidated_document(doc: ConsolidatedDocument, output_dir: str) -> str:
    """Save consolidated document and chunks for a company."""
    company_dir = os.path.join(output_dir, doc.company)
    os.makedirs(company_dir, exist_ok=True)
    
    # Save full combined text
    combined_text_path = os.path.join(company_dir, "_COMBINED_DOCUMENT.txt")
    with open(combined_text_path, 'w', encoding='utf-8') as f:
        f.write(doc.full_text)
    
    # Save chunks
    chunks_path = os.path.join(company_dir, "_COMBINED_CHUNKS.json")
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(doc.chunks, f, indent=2)
    
    # Save metadata
    meta = {
        "company": doc.company,
        "source_files": doc.source_files,
        "total_chars": doc.total_chars,
        "total_tokens": doc.total_tokens,
        "num_chunks": len(doc.chunks),
        "source_references": [
            {
                "file": r.file_name,
                "char_range": [r.start_char, r.end_char],
                "chunk_range": [r.start_chunk, r.end_chunk]
            }
            for r in doc.source_references
        ]
    }
    meta_path = os.path.join(company_dir, "_COMBINED_META.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"  Saved consolidated document to {company_dir}")
    
    return chunks_path


def get_all_companies(manifest: List[Dict]) -> List[str]:
    """Get unique company names from manifest."""
    companies = list(set(m['company'] for m in manifest))
    return sorted(companies)


# =============================================================================
# ITERATIVE FULL-DOCUMENT EXTRACTION
# =============================================================================

def create_extraction_plan(doc: ConsolidatedDocument, max_chunks_per_call: int = 6) -> List[List[int]]:
    """
    Create a plan to ensure ALL chunks are processed.
    
    Returns list of chunk ID groups to process.
    Each group should have overlap with previous for context continuity.
    """
    total_chunks = len(doc.chunks)
    
    if total_chunks <= max_chunks_per_call:
        # All chunks fit in one call
        return [[c['id'] for c in doc.chunks]]
    
    # Create overlapping groups
    groups = []
    step = max_chunks_per_call - 2  # 2 chunks overlap between groups
    
    for start in range(0, total_chunks, step):
        end = min(start + max_chunks_per_call, total_chunks)
        group = list(range(start, end))
        groups.append(group)
        
        if end >= total_chunks:
            break
    
    logger.info(f"  Extraction plan: {len(groups)} groups for {total_chunks} chunks")
    
    return groups


def get_chunks_for_extraction(doc: ConsolidatedDocument, chunk_ids: List[int]) -> str:
    """Get combined text for a set of chunks."""
    texts = []
    for chunk in doc.chunks:
        if chunk['id'] in chunk_ids:
            # Add source info header
            sources = chunk.get('source_files', [])
            if sources:
                source_note = f"[From: {', '.join(sources)}]"
            else:
                source_note = ""
            
            texts.append(f"{source_note}\n{chunk['text']}")
    
    return "\n\n---\n\n".join(texts)


def merge_extraction_results(results: List[Dict], field_groups: Dict) -> Dict:
    """
    Merge extraction results from multiple chunk groups.
    
    Strategy:
    - For each field, keep the value with highest confidence
    - Combine evidence from all sources
    - Track which chunks contributed
    """
    merged = {}
    
    for group_name in field_groups.keys():
        merged[group_name] = {"fields": {}}
        
        for result in results:
            if group_name not in result:
                continue
            
            group_data = result[group_name]
            if 'fields' not in group_data:
                continue
            
            for field_name, field_data in group_data['fields'].items():
                if not isinstance(field_data, dict):
                    continue
                
                value = field_data.get('value', 'NOT_FOUND')
                confidence = field_data.get('confidence', 'LOW')
                evidence = field_data.get('evidence', '')
                
                # Skip NOT_FOUND / POSSIBLY_PRESENT values
                if value in ['NOT_FOUND', 'N/A', '', None, 'POSSIBLY_PRESENT']:
                    if field_name not in merged[group_name]['fields']:
                        merged[group_name]['fields'][field_name] = field_data
                    continue
                
                # Compare with existing
                existing = merged[group_name]['fields'].get(field_name)
                
                if not existing or existing.get('value') in ['NOT_FOUND', 'N/A', '', None]:
                    # No existing value, use this one
                    merged[group_name]['fields'][field_name] = field_data
                else:
                    # Compare confidence
                    conf_rank = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
                    existing_rank = conf_rank.get(existing.get('confidence', 'LOW'), 0)
                    new_rank = conf_rank.get(confidence, 0)
                    
                    if new_rank > existing_rank:
                        # New value has higher confidence
                        merged[group_name]['fields'][field_name] = field_data
                    elif new_rank == existing_rank and evidence:
                        # Same confidence, append evidence
                        existing_evidence = existing.get('evidence', '')
                        if evidence not in existing_evidence:
                            merged[group_name]['fields'][field_name]['evidence'] = \
                                f"{existing_evidence} | {evidence}"
    
    return merged
