"""
=================================================
Deep Field Extraction: Field-by-Field Fallback
=================================================
For critical fields that were missed in the first pass,
this module does targeted single-field extraction with
multiple retrieval strategies.
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple

from .config import FIELD_GROUPS, EXTRACTABLE_FIELDS
from .retriever import BM25Retriever

logger = logging.getLogger(__name__)


# =====================================================================
# FIELD-SPECIFIC SEARCH STRATEGIES
# =====================================================================

FIELD_SEARCH_STRATEGIES = {
    # Basic Info
    "Borrower name": {
        "primary_keywords": ["borrower", "company", "obligor", "debtor"],
        "context_patterns": [r'("the borrower"|"borrower means"|borrower:?\s*)', r'("the company"|company name)'],
        "value_pattern": r'(?:borrower|company|obligor)[\s:]*([A-Z][A-Za-z\s&.,]+(?:Ltd|LLC|Inc|Corp|B\.V\.|S\.A\.|GmbH|Plc)?)',
    },
    "Credit limit ['000]": {
        "primary_keywords": ["credit", "limit", "facility", "amount", "commitment", "aggregate"],
        "context_patterns": [r'(credit limit|facility amount|aggregate commitment|total commitment)'],
        "value_pattern": r'(?:EUR|USD|GBP|\$|€|£)\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|m|bn|billion))?',
    },
    "Spread [%]": {
        "primary_keywords": ["spread", "margin", "basis points", "bps", "plus"],
        "context_patterns": [r'(applicable margin|interest margin|margin means|spread of)'],
        "value_pattern": r'(\d+(?:\.\d+)?)\s*(?:%|percent|basis points|bps)',
    },
    "Maturity date [MM/YYYY] of the facility": {
        "primary_keywords": ["maturity", "termination", "final", "expiry", "repayment date"],
        "context_patterns": [r'(final maturity|termination date|maturity date|facility ends)'],
        "value_pattern": r'(\d{1,2}[\/\-]\d{4}|\d{4}[\/\-]\d{1,2}|\w+\s+\d{4})',
    },
    "Base Rate of the facility": {
        "primary_keywords": ["EURIBOR", "SOFR", "LIBOR", "base rate", "reference rate", "benchmark"],
        "context_patterns": [r'(reference rate|base rate|benchmark rate)'],
        "value_pattern": r'(EURIBOR|SOFR|LIBOR|SONIA|ESTR|Term SOFR)',
    },
    "Inception date [MM/YYYY] of the facility": {
        "primary_keywords": ["inception", "signing", "effective date", "commencement", "dated"],
        "context_patterns": [r'(effective date|signing date|dated as of|entered into)'],
        "value_pattern": r'(\d{1,2}[\/\-]\d{4}|\d{4}[\/\-]\d{1,2}|\w+\s+\d{4})',
    },
    "Project location [Country]": {
        "primary_keywords": ["location", "country", "jurisdiction", "situated", "located"],
        "context_patterns": [r'(located in|situated in|project location|jurisdiction)'],
        "value_pattern": r'(Spain|Germany|France|Netherlands|UK|United Kingdom|Italy|Poland|Belgium|Portugal|Ireland|Greece|Austria|Sweden|Denmark|Norway|Finland|Czech|Romania|Hungary|Bulgaria|Croatia|Slovakia|Slovenia|Luxembourg|Estonia|Latvia|Lithuania|Cyprus|Malta)',
    },
    "Covenant leading to dividend lock-up: Backward-looking DSCR (B-DSCR)": {
        "primary_keywords": ["DSCR", "debt service coverage", "backward", "historical", "covenant"],
        "context_patterns": [r'(historical DSCR|backward.looking DSCR|DSCR covenant|debt service coverage ratio)'],
        "value_pattern": r'(\d+(?:\.\d+)?)[:\s]*(?:1|x|times)',
    },
    "ING share [%]": {
        "primary_keywords": ["ING", "commitment", "share", "participation", "proportion"],
        "context_patterns": [r'(ING Bank|ING\'s commitment|ING participation)'],
        "value_pattern": r'(\d+(?:\.\d+)?)\s*%',
    },
}


# =====================================================================
# TARGETED EXTRACTION PROMPTS
# =====================================================================

SINGLE_FIELD_PROMPT = """You are extracting ONE specific data point from a financial document.

FIELD TO EXTRACT: {field_name}
DESCRIPTION: {field_description}

CONTEXT PASSAGES:
---
{context}
---

INSTRUCTIONS:
1. Find the exact value for this field in the passages above
2. Provide the EXACT quote that contains this information
3. If the field is not mentioned, respond with NOT_FOUND

Respond in this exact JSON format:
{{
    "value": "the extracted value or NOT_FOUND",
    "evidence": "exact quote from document (max 200 chars)",
    "confidence": "HIGH if explicitly stated, MEDIUM if inferred, LOW if uncertain",
    "reasoning": "brief explanation of how you found this"
}}

IMPORTANT: Only extract if explicitly stated. Do not guess or infer."""


def get_field_description(field_name: str) -> str:
    """Get description for a field from FIELD_GROUPS."""
    for group_name, group_config in FIELD_GROUPS.items():
        for fname, fdesc in group_config['fields']:
            if fname == field_name:
                return fdesc
    return field_name


# =====================================================================
# PATTERN-BASED PRE-EXTRACTION
# =====================================================================

def extract_with_patterns(text: str, field_name: str) -> List[str]:
    """
    Use regex patterns to find potential values before LLM extraction.
    Returns list of candidate values.
    """
    if field_name not in FIELD_SEARCH_STRATEGIES:
        return []
    
    strategy = FIELD_SEARCH_STRATEGIES[field_name]
    candidates = []
    
    # Find context matches
    for pattern in strategy.get('context_patterns', []):
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            # Get surrounding text
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 200)
            context = text[start:end]
            
            # Look for value pattern
            value_pattern = strategy.get('value_pattern')
            if value_pattern:
                value_matches = re.findall(value_pattern, context, re.IGNORECASE)
                candidates.extend(value_matches)
    
    return list(set(candidates))


def find_best_context_for_field(retriever: BM25Retriever, 
                                field_name: str,
                                top_k: int = 5) -> List[str]:
    """
    Find the best context chunks for a specific field.
    Uses multiple search strategies.
    """
    contexts = []
    
    # Strategy 1: Use field-specific keywords
    if field_name in FIELD_SEARCH_STRATEGIES:
        keywords = FIELD_SEARCH_STRATEGIES[field_name]['primary_keywords']
        results = retriever.search_with_keywords(keywords, top_k)
        contexts.extend([text for _, _, text in results])
    
    # Strategy 2: Use field name directly
    results = retriever.search(field_name, top_k)
    for _, _, text in results:
        if text not in contexts:
            contexts.append(text)
    
    # Strategy 3: Use description
    description = get_field_description(field_name)
    if description != field_name:
        results = retriever.search(description, top_k // 2)
        for _, _, text in results:
            if text not in contexts:
                contexts.append(text)
    
    return contexts[:top_k]


# =====================================================================
# DEEP EXTRACTION
# =====================================================================

def deep_extract_field(field_name: str, retriever: BM25Retriever,
                      model, tokenizer, llm_generate_fn) -> Dict:
    """
    Perform deep extraction for a single field.
    Uses multiple strategies to maximize recall.
    """
    from .extract_v2 import parse_json_response
    
    # Get best context
    contexts = find_best_context_for_field(retriever, field_name, top_k=6)
    
    if not contexts:
        return {
            "value": "NOT_FOUND",
            "evidence": "",
            "confidence": "LOW",
            "reasoning": "No relevant context found"
        }
    
    # Pre-extraction with patterns
    full_context = '\n\n'.join(contexts)
    pattern_candidates = extract_with_patterns(full_context, field_name)
    
    # Build prompt with pattern hints
    field_description = get_field_description(field_name)
    
    prompt_context = full_context
    if pattern_candidates:
        prompt_context += f"\n\n[HINT: Possible values found by pattern matching: {', '.join(pattern_candidates[:5])}]"
    
    prompt = SINGLE_FIELD_PROMPT.format(
        field_name=field_name,
        field_description=field_description,
        context=prompt_context[:8000]  # Limit context size
    )
    
    messages = [
        {"role": "system", "content": "You are a precise financial data extractor. Respond only with valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    response = llm_generate_fn(messages, model, tokenizer, max_tokens=500)
    result = parse_json_response(response)
    
    if result:
        return result
    
    # Fallback: use pattern results if LLM failed
    if pattern_candidates:
        return {
            "value": pattern_candidates[0],
            "evidence": f"Pattern-matched from document",
            "confidence": "MEDIUM",
            "reasoning": "Extracted via pattern matching, LLM verification failed"
        }
    
    return {
        "value": "NOT_FOUND",
        "evidence": "",
        "confidence": "LOW",
        "reasoning": "Extraction failed"
    }


def deep_extract_missing_fields(current_extraction: Dict,
                               retriever: BM25Retriever,
                               model, tokenizer, 
                               llm_generate_fn) -> Dict:
    """
    Find fields marked as NOT_FOUND and try to extract them again
    with more targeted approaches.
    """
    # Find missing/low-confidence fields
    missing_fields = []
    
    for group_name, group_data in current_extraction.items():
        if 'fields' not in group_data:
            continue
        
        for field_name, field_data in group_data['fields'].items():
            if isinstance(field_data, dict):
                value = field_data.get('value', '')
                confidence = field_data.get('confidence', 'LOW')
                
                if value in ['NOT_FOUND', 'N/A', 'EXTRACTION_ERROR', ''] or confidence == 'LOW':
                    missing_fields.append((field_name, group_name))
    
    if not missing_fields:
        logger.info("  No missing fields to deep extract")
        return current_extraction
    
    logger.info(f"  Deep extracting {len(missing_fields)} missing/low-confidence fields")
    
    # Deep extract each missing field
    for field_name, group_name in missing_fields:
        logger.info(f"    Deep extracting: {field_name}")
        
        result = deep_extract_field(field_name, retriever, model, tokenizer, llm_generate_fn)
        
        # Update if we found something better
        if result.get('value') not in ['NOT_FOUND', 'N/A', '']:
            if group_name in current_extraction and 'fields' in current_extraction[group_name]:
                current_extraction[group_name]['fields'][field_name] = result
                logger.info(f"      Found: {result.get('value')[:50]}...")
    
    return current_extraction


# =====================================================================
# TABLE EXTRACTION
# =====================================================================

def extract_from_tables(text: str, field_name: str) -> Optional[str]:
    """
    Specialized extraction for tabular data.
    Many financial documents have key info in tables.
    """
    # Find table-like structures
    table_patterns = [
        r'\|[^|]+\|',  # Pipe-delimited
        r'(?:^|\n)([^\t]+\t[^\t]+\t[^\t]+)',  # Tab-delimited
    ]
    
    # Field name variations to search for
    field_variants = [
        field_name.lower(),
        field_name.replace(' ', ''),
        re.sub(r'[^\w\s]', '', field_name).lower(),
    ]
    
    for line in text.split('\n'):
        for variant in field_variants:
            if variant in line.lower():
                # Try to extract value from same line
                parts = re.split(r'[|:\t]', line)
                if len(parts) >= 2:
                    for i, part in enumerate(parts):
                        if variant in part.lower() and i + 1 < len(parts):
                            value = parts[i + 1].strip()
                            if value and value.lower() not in ['', 'n/a', 'nan']:
                                return value
    
    return None


# =====================================================================
# CROSS-REFERENCE VALIDATION
# =====================================================================

def cross_reference_fields(extractions: Dict) -> Dict:
    """
    Check consistency between related fields.
    E.g., if facility is syndicated, ING share should be specified.
    """
    issues = []
    
    # Get all fields flat
    all_fields = {}
    for group_data in extractions.values():
        if 'fields' in group_data:
            for fname, fdata in group_data['fields'].items():
                if isinstance(fdata, dict):
                    all_fields[fname] = fdata.get('value', '')
                else:
                    all_fields[fname] = str(fdata)
    
    # Rule checks
    if all_fields.get('Is facility syndicated?', '').lower() == 'yes':
        if all_fields.get('ING share [%]', '') in ['NOT_FOUND', 'N/A', '']:
            issues.append("Facility is syndicated but ING share not found")
    
    if all_fields.get('Hedging: is the facility hedged?', '').lower() == 'yes':
        hedging_fields = ['Hedging: how is the facility hedged', 'Hedging: notional']
        missing_hedging = [f for f in hedging_fields if all_fields.get(f, '') in ['NOT_FOUND', 'N/A', '']]
        if missing_hedging:
            issues.append(f"Hedging is yes but missing: {', '.join(missing_hedging)}")
    
    if all_fields.get('Readiness [In operation / Construction needed]', '').lower() == 'construction needed':
        if all_fields.get('If construction needed: Readiness year', '') in ['NOT_FOUND', 'N/A', '']:
            issues.append("Construction needed but readiness year not found")
    
    return {"consistency_issues": issues}
