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
                      model, tokenizer, llm_generate_fn,
                      full_text: str = None) -> Dict:
    """
    Perform deep extraction for a single field.
    Uses multiple strategies to maximize recall:
    1. BM25 retrieval with multiple strategies
    2. Table-specific extraction
    3. Regex pattern pre-matching
    4. LLM extraction with pattern hints
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

    # Strategy: Try table extraction (many financial fields live in tables)
    table_value = None
    if full_text:
        table_value = extract_from_tables(full_text, field_name)
    if not table_value:
        # Also try table extraction on the retrieved context chunks
        table_value = extract_from_tables(full_context, field_name)

    if table_value and table_value not in pattern_candidates:
        pattern_candidates.insert(0, table_value)

    # Build prompt with pattern hints
    field_description = get_field_description(field_name)

    prompt_context = full_context
    if pattern_candidates:
        prompt_context += f"\n\n[HINT: Possible values found by pattern matching: {', '.join(pattern_candidates[:5])}]"

    # Token-based truncation to fit within input budget
    max_context_tokens = 6000
    ctx_tokens = tokenizer.encode(prompt_context, add_special_tokens=False)
    if len(ctx_tokens) > max_context_tokens:
        prompt_context = tokenizer.decode(ctx_tokens[:max_context_tokens], skip_special_tokens=True)

    prompt = SINGLE_FIELD_PROMPT.format(
        field_name=field_name,
        field_description=field_description,
        context=prompt_context
    )

    messages = [
        {"role": "system", "content": "You are a precise financial data extractor. Respond only with valid JSON."},
        {"role": "user", "content": prompt}
    ]

    response = llm_generate_fn(messages, model, tokenizer, max_tokens=500,
                               step_name=f"deep_extract_{field_name[:40]}")
    result = parse_json_response(response)

    if result:
        return result

    # Fallback: use pattern results if LLM failed
    if pattern_candidates:
        return {
            "value": pattern_candidates[0],
            "evidence": "Pattern-matched from document",
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
                               llm_generate_fn,
                               full_text: str = None) -> Dict:
    """
    Find fields marked as NOT_FOUND, POSSIBLY_PRESENT, or low-confidence
    and try to extract them again with more targeted approaches.

    POSSIBLY_PRESENT is a signal from the primary extraction that the
    field likely exists in the document but wasn't visible in the chunks
    that were provided. These are prioritized over generic NOT_FOUND.
    """
    # Find missing/low-confidence fields, prioritizing POSSIBLY_PRESENT
    possibly_present_fields = []
    missing_fields = []

    for group_name, group_data in current_extraction.items():
        if 'fields' not in group_data:
            continue

        for field_name, field_data in group_data['fields'].items():
            if isinstance(field_data, dict):
                value = field_data.get('value', '')
                confidence = field_data.get('confidence', 'LOW')

                if value == 'POSSIBLY_PRESENT':
                    possibly_present_fields.append((field_name, group_name))
                elif value in ['NOT_FOUND', 'N/A', 'EXTRACTION_ERROR', ''] or confidence == 'LOW':
                    missing_fields.append((field_name, group_name))

    # Process POSSIBLY_PRESENT first (higher chance of success)
    all_targets = possibly_present_fields + missing_fields

    if not all_targets:
        logger.info("  No missing fields to deep extract")
        return current_extraction

    logger.info(f"  Deep extracting {len(all_targets)} fields "
                f"({len(possibly_present_fields)} POSSIBLY_PRESENT, "
                f"{len(missing_fields)} NOT_FOUND/LOW)")

    # Deep extract each missing field
    for field_name, group_name in all_targets:
        logger.info(f"    Deep extracting: {field_name}")

        result = deep_extract_field(
            field_name, retriever, model, tokenizer, llm_generate_fn,
            full_text=full_text
        )

        # Update if we found something better
        if result.get('value') not in ['NOT_FOUND', 'N/A', '', 'POSSIBLY_PRESENT']:
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
    Check consistency between related fields and auto-fix simple contradictions.
    E.g., if facility is syndicated, ING share should be specified.

    Returns dict with:
      - consistency_issues: list of warning strings
      - auto_fixes: dict of field_name -> corrected value (applied in-place)
    """
    issues = []
    auto_fixes = {}
    skip_values = {'NOT_FOUND', 'N/A', '', None, 'POSSIBLY_PRESENT', 'EXTRACTION_ERROR'}

    # Get all fields flat
    all_fields = {}
    for group_data in extractions.values():
        if 'fields' in group_data:
            for fname, fdata in group_data['fields'].items():
                if isinstance(fdata, dict):
                    all_fields[fname] = fdata.get('value', '')
                else:
                    all_fields[fname] = str(fdata)

    def _val(field_name: str) -> str:
        """Get field value, empty string if missing/not_found."""
        v = all_fields.get(field_name, '')
        return '' if v in skip_values else v

    def _is_yes(field_name: str) -> bool:
        return _val(field_name).strip().lower() == 'yes'

    def _is_no(field_name: str) -> bool:
        return _val(field_name).strip().lower() == 'no'

    # ─── Rule 1: Syndication ↔ ING share ───
    if _is_yes('Is facility syndicated?'):
        if not _val('ING share [%]'):
            issues.append("Facility is syndicated but ING share not found")

    # ─── Rule 2: Hedging consistency ───
    if _is_yes('Hedging: is the facility hedged?'):
        hedging_fields = ['Hedging: how is the facility hedged', 'Hedging: notional']
        missing_hedging = [f for f in hedging_fields if not _val(f)]
        if missing_hedging:
            issues.append(f"Hedging is yes but missing: {', '.join(missing_hedging)}")
    elif _is_no('Hedging: is the facility hedged?'):
        # If hedging is No but hedging details are present → suspicious
        hedging_detail_fields = [
            'Hedging: how is the facility hedged', 'Hedging: fixed rate',
            'Hedging: notional', 'Hedging: spread'
        ]
        found_details = [f for f in hedging_detail_fields if _val(f)]
        if found_details:
            issues.append(f"Hedging is No but found details: {', '.join(found_details)} — may need correction to Yes")

    # ─── Rule 3: Construction readiness ───
    if 'construction' in _val('Readiness [In operation / Construction needed]').lower():
        if not _val('If construction needed: Readiness year'):
            issues.append("Construction needed but readiness year not found")
    elif 'operation' in _val('Readiness [In operation / Construction needed]').lower():
        # In operation → construction fields should be NOT_FOUND, not filled
        if _val('If construction needed: Readiness year'):
            issues.append("In operation but construction readiness year is filled — likely incorrect")

    # ─── Rule 4: Covenants Yes/No ↔ covenant details ───
    if _is_yes('If covenants specified?'):
        # At least one ratio should be present
        ratio_fields = [
            'Covenant leading to dividend lock-up: Backward-looking DSCR (B-DSCR)',
            'Covenant leading to dividend lock-up: Forward-looking DSCR (F-DSCR)',
            'Covenant leading to dividend lock-up: Loan Life Coverage Ratio (LLCR)',
            'Covenant leading to dividend lock-up: Interest Coverage Ratio (ICR)',
            'Covenant leading to dividend lock-up: Net Debt / EBITDA',
        ]
        found_ratios = [f for f in ratio_fields if _val(f)]
        if not found_ratios:
            issues.append("Covenants=Yes but no specific covenant ratios found")
    elif _is_no('If covenants specified?'):
        # If No but ratios found → auto-fix to Yes
        ratio_fields = [
            'Covenant leading to dividend lock-up: Backward-looking DSCR (B-DSCR)',
            'Covenant leading to dividend lock-up: Forward-looking DSCR (F-DSCR)',
            'Covenant leading to dividend lock-up: Loan Life Coverage Ratio (LLCR)',
            'Covenant leading to dividend lock-up: Interest Coverage Ratio (ICR)',
        ]
        found_ratios = [f for f in ratio_fields if _val(f)]
        if found_ratios:
            issues.append(f"Covenants=No but found ratios: {found_ratios[0]} — auto-fixing to Yes")
            auto_fixes['If covenants specified?'] = 'Yes'

    # ─── Rule 5: D&C Contractor consistency ───
    if _is_yes('If D&C Contractor specified?'):
        if not _val('Name of the D&C Contractor'):
            issues.append("D&C Contractor=Yes but name not found")
    elif _is_no('If D&C Contractor specified?'):
        if _val('Name of the D&C Contractor'):
            issues.append("D&C Contractor=No but contractor name is filled — auto-fixing to Yes")
            auto_fixes['If D&C Contractor specified?'] = 'Yes'

    # ─── Rule 6: Completion guarantee consistency ───
    if _is_yes('If completion guarantees specified?'):
        if not _val('Guarantee Development sponsors name'):
            issues.append("Completion guarantees=Yes but guarantor name not found")
    elif _is_no('If completion guarantees specified?'):
        if _val('Guarantee Development sponsors name'):
            issues.append("Completion guarantees=No but guarantor name is filled — auto-fixing to Yes")
            auto_fixes['If completion guarantees specified?'] = 'Yes'

    # ─── Rule 7: Revenue mitigating factors consistency ───
    if _is_yes('If revenue mitigating factors specified?'):
        if not _val('Type of mitigating factor'):
            issues.append("Revenue mitigating factors=Yes but type not found")
    elif _is_no('If revenue mitigating factors specified?'):
        if _val('Type of mitigating factor') or _val('Contractual or regulatory factor guarantor'):
            issues.append("Revenue mitigating=No but details found — auto-fixing to Yes")
            auto_fixes['If revenue mitigating factors specified?'] = 'Yes'

    # ─── Rule 8: Sponsor consistency ───
    if _is_yes('Is a Sponsor linked to the project?'):
        if not _val('Sponsor Name'):
            issues.append("Sponsor=Yes but sponsor name not found")
    elif _is_no('Is a Sponsor linked to the project?'):
        if _val('Sponsor Name'):
            issues.append("Sponsor=No but sponsor name found — auto-fixing to Yes")
            auto_fixes['Is a Sponsor linked to the project?'] = 'Yes'

    # ─── Rule 9: HoldCo consistency ───
    if _is_yes('Is a HoldCo linked to the project?'):
        if not _val('Name of the HoldCo linked to the project'):
            issues.append("HoldCo=Yes but HoldCo name not found")
    elif _is_no('Is a HoldCo linked to the project?'):
        if _val('Name of the HoldCo linked to the project'):
            issues.append("HoldCo=No but HoldCo name found — auto-fixing to Yes")
            auto_fixes['Is a HoldCo linked to the project?'] = 'Yes'

    # ─── Rule 10: Cash sweep consistency ───
    if _is_yes('Is there a cash sweep mechanism applicable to the facility?'):
        if not _val('Cash sweep structure'):
            issues.append("Cash sweep=Yes but structure not found")
    elif _is_no('Is there a cash sweep mechanism applicable to the facility?'):
        if _val('Cash sweep structure'):
            issues.append("Cash sweep=No but structure found — auto-fixing to Yes")
            auto_fixes['Is there a cash sweep mechanism applicable to the facility?'] = 'Yes'

    # ─── Rule 11: Interest rate type consistency ───
    base_rate = _val('Base Rate of the facility')
    fix_rate = _val('Fix interest rate [%]')
    spread = _val('Spread [%]')
    if base_rate and fix_rate:
        issues.append(f"Both base rate ({base_rate}) and fixed rate ({fix_rate}) found — usually only one applies")
    if base_rate and not spread:
        issues.append("Base rate found but spread is missing — floating rate usually has a spread/margin")

    # ─── Rule 12: Date sanity ───
    inception = _val('Inception date [MM/YYYY] of the facility')
    maturity = _val('Maturity date [MM/YYYY] of the facility')
    if inception and maturity:
        # Simple year check
        inc_year = re.search(r'(\d{4})', inception)
        mat_year = re.search(r'(\d{4})', maturity)
        if inc_year and mat_year:
            if int(mat_year.group(1)) < int(inc_year.group(1)):
                issues.append(f"Maturity ({maturity}) is before inception ({inception}) — dates may be swapped")

    # ─── Apply auto-fixes to extractions ───
    if auto_fixes:
        for group_data in extractions.values():
            if 'fields' not in group_data:
                continue
            for fname, fdata in group_data['fields'].items():
                if fname in auto_fixes and isinstance(fdata, dict):
                    old_val = fdata.get('value', '')
                    fdata['value'] = auto_fixes[fname]
                    fdata['evidence'] = fdata.get('evidence', '') + f" [AUTO-FIXED from {old_val}]"
                    fdata['confidence'] = 'MEDIUM'
                    logger.info(f"    Auto-fixed: {fname}: {old_val} → {auto_fixes[fname]}")

    return {"consistency_issues": issues, "auto_fixes": auto_fixes}
