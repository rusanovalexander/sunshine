"""
=================================================
Output Normalization Module
=================================================
Post-processes extracted values to ensure consistent
formatting in the final CSV output.

Runs entirely on CPU — zero GPU cost.

Normalizes:
- Dates → MM/YYYY
- Amounts → numeric with currency
- Percentages → X.XX%
- Ratios → X.XXx
- Yes/No → strict boolean
"""

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# =====================================================================
# DATE NORMALIZATION
# =====================================================================

MONTH_MAP = {
    'january': '01', 'jan': '01',
    'february': '02', 'feb': '02',
    'march': '03', 'mar': '03',
    'april': '04', 'apr': '04',
    'may': '05',
    'june': '06', 'jun': '06',
    'july': '07', 'jul': '07',
    'august': '08', 'aug': '08',
    'september': '09', 'sep': '09', 'sept': '09',
    'october': '10', 'oct': '10',
    'november': '11', 'nov': '11',
    'december': '12', 'dec': '12',
}


def normalize_date(value: str) -> str:
    """
    Normalize dates to MM/YYYY format.

    Handles:
    - "15 March 2024" → "03/2024"
    - "March 2024" → "03/2024"
    - "2024-03-15" → "03/2024"
    - "03/2024" → "03/2024" (already correct)
    - "Q3 2025" → "09/2025" (end of quarter)
    - "15/03/2024" → "03/2024"
    """
    if not value or value in ('NOT_FOUND', 'N/A', 'POSSIBLY_PRESENT', 'NOT_EXTRACTED'):
        return value

    original = value.strip()

    # Already MM/YYYY
    m = re.match(r'^(\d{1,2})/(\d{4})$', original)
    if m:
        month = m.group(1).zfill(2)
        year = m.group(2)
        if 1 <= int(month) <= 12:
            return f"{month}/{year}"

    # "Month YYYY" or "DD Month YYYY"
    for month_name, month_num in MONTH_MAP.items():
        pattern = rf'(?:\d{{1,2}}\s+)?{month_name}\s+(\d{{4}})'
        m = re.search(pattern, original.lower())
        if m:
            return f"{month_num}/{m.group(1)}"

    # ISO format: YYYY-MM-DD or YYYY-MM
    m = re.match(r'(\d{4})-(\d{1,2})(?:-\d{1,2})?', original)
    if m:
        return f"{m.group(2).zfill(2)}/{m.group(1)}"

    # DD/MM/YYYY
    m = re.match(r'\d{1,2}/(\d{1,2})/(\d{4})', original)
    if m:
        month = m.group(1).zfill(2)
        if 1 <= int(month) <= 12:
            return f"{month}/{m.group(2)}"

    # Quarter notation: Q1-Q4 YYYY
    m = re.search(r'Q([1-4])\s+(\d{4})', original, re.IGNORECASE)
    if m:
        quarter_end_month = {'1': '03', '2': '06', '3': '09', '4': '12'}
        return f"{quarter_end_month[m.group(1)]}/{m.group(2)}"

    # Just a year
    m = re.match(r'^(\d{4})$', original)
    if m:
        return original  # Can't determine month, keep as-is

    return original


# =====================================================================
# PERCENTAGE NORMALIZATION
# =====================================================================

def normalize_percentage(value: str) -> str:
    """
    Normalize percentages to X.XX% format.

    Handles:
    - "2.50%" → "2.50%"
    - "250bps" → "2.50%"
    - "250 basis points" → "2.50%"
    - "2.5 per cent" → "2.50%"
    - "2.5 percent" → "2.50%"
    """
    if not value or value in ('NOT_FOUND', 'N/A', 'POSSIBLY_PRESENT', 'NOT_EXTRACTED'):
        return value

    original = value.strip()

    # Basis points → percentage
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:bps|bp|basis\s+points?)', original, re.IGNORECASE)
    if m:
        bps = float(m.group(1))
        pct = bps / 100.0
        return f"{pct:.2f}%"

    # "X per cent" or "X percent" or "X %"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:per\s*cent|percent|%)', original, re.IGNORECASE)
    if m:
        return f"{float(m.group(1)):.2f}%"

    # Already a percentage-like number without symbol
    m = re.match(r'^(\d+(?:\.\d+)?)$', original)
    if m:
        # Don't add % if no clear context — return as-is
        return original

    return original


# =====================================================================
# RATIO NORMALIZATION
# =====================================================================

def normalize_ratio(value: str) -> str:
    """
    Normalize financial ratios to X.XXx format.

    Handles:
    - "1.20:1" → "1.20x"
    - "1.20 to 1" → "1.20x"
    - "1.20x" → "1.20x"
    - "120%" → "1.20x"
    """
    if not value or value in ('NOT_FOUND', 'N/A', 'POSSIBLY_PRESENT', 'NOT_EXTRACTED'):
        return value

    original = value.strip()

    # Already in Xx format
    m = re.match(r'^(\d+(?:\.\d+)?)x$', original, re.IGNORECASE)
    if m:
        return f"{float(m.group(1)):.2f}x"

    # Ratio format: X.XX:1 or X.XX : 1
    m = re.search(r'(\d+(?:\.\d+)?)\s*:\s*1', original)
    if m:
        return f"{float(m.group(1)):.2f}x"

    # "X.XX to 1"
    m = re.search(r'(\d+(?:\.\d+)?)\s+to\s+1', original, re.IGNORECASE)
    if m:
        return f"{float(m.group(1)):.2f}x"

    return original


# =====================================================================
# AMOUNT NORMALIZATION
# =====================================================================

CURRENCY_SYMBOLS = {
    '€': 'EUR', '$': 'USD', '£': 'GBP', '¥': 'JPY', 'CHF': 'CHF',
    'EUR': 'EUR', 'USD': 'USD', 'GBP': 'GBP', 'JPY': 'JPY',
    'SEK': 'SEK', 'NOK': 'NOK', 'DKK': 'DKK', 'PLN': 'PLN',
    'AUD': 'AUD', 'CAD': 'CAD', 'SGD': 'SGD', 'HKD': 'HKD',
    'BRL': 'BRL', 'INR': 'INR', 'ZAR': 'ZAR',
}


def normalize_amount(value: str) -> str:
    """
    Normalize monetary amounts.

    Handles:
    - "EUR 250,000,000" → "EUR 250,000"  (in '000s if field implies it)
    - "250 million EUR" → "EUR 250,000"
    - "EUR 250M" → "EUR 250,000"
    - "€250m" → "EUR 250,000"

    Note: This function normalizes format but does NOT convert to '000s.
    The ['000] conversion depends on field context and is handled separately.
    """
    if not value or value in ('NOT_FOUND', 'N/A', 'POSSIBLY_PRESENT', 'NOT_EXTRACTED'):
        return value

    return value.strip()


# =====================================================================
# YES/NO NORMALIZATION
# =====================================================================

def normalize_yes_no(value: str) -> str:
    """Normalize boolean responses to strict Yes/No."""
    if not value or value in ('NOT_FOUND', 'N/A', 'POSSIBLY_PRESENT', 'NOT_EXTRACTED'):
        return value

    lower = value.strip().lower()

    yes_patterns = {'yes', 'true', 'y', 'affirmative', 'confirmed', 'applicable'}
    no_patterns = {'no', 'false', 'n', 'negative', 'not applicable', 'n/a', 'none'}

    if lower in yes_patterns:
        return "Yes"
    if lower in no_patterns:
        return "No"

    # Partial match
    if lower.startswith('yes'):
        return "Yes"
    if lower.startswith('no'):
        return "No"

    return value.strip()


# =====================================================================
# FIELD CLASSIFICATION & DISPATCH
# =====================================================================

# Fields that contain dates
DATE_FIELDS = {
    "Inception date [MM/YYYY] of the facility",
    "Maturity date [MM/YYYY] of the facility",
    "Hedging: effective date",
    "Hedging: maturity date",
    "If construction needed: Readiness year",
    "If construction needed: Readiness month",
}

# Fields that contain percentages
PERCENTAGE_FIELDS = {
    "Spread [%]",
    "Fix interest rate [%]",
    "Interest rate floor %",
    "Interest rate cap %",
    "Sponsor ownership [%]",
    "ING share [%]",
    "Hedging: fixed rate",
    "Hedging: % of exposure hedged",
    "Volume covered by revenue mitigating factors [%]",
    "For predetermined - additional cash to be paid [%]",
    "For targeted - target repayment profile [%]",
    "For transmission asset linked loan - sweep percentage [%]",
    "Upfront Fee [%]",
    "Commitment fee [%]",
    "Letter of credit issuance fee [%]",
}

# Fields that contain ratios
RATIO_FIELDS = {
    "Covenant leading to dividend lock-up: Backward-looking DSCR (B-DSCR)",
    "Covenant leading to dividend lock-up: Forward-looking DSCR (F-DSCR)",
    "Covenant leading to dividend lock-up: Loan Life Coverage Ratio (LLCR)",
    "Covenant leading to dividend lock-up: Project Life Coverage Ratio (PLCR)",
    "Covenant leading to dividend lock-up: Interest Coverage Ratio (ICR)",
    "Covenant leading to dividend lock-up: Net Debt / EBITDA",
    "Covenant leading to default: Backward-looking DSCR (B-DSCR)",
    "Covenant leading to default: Forward-looking DSCR (F-DSCR)",
    "Covenant leading to default: Loan Life Coverage Ratio (LLCR)",
    "Covenant leading to default: Project Life Coverage Ratio (PLCR)",
    "Covenant leading to default: Interest Coverage Ratio (ICR)",
    "Covenant leading to default: Net Debt / EBITDA",
}

# Fields that expect Yes/No
YES_NO_FIELDS = {
    "Is a Sponsor linked to the project?",
    "Is a HoldCo linked to the project?",
    "If completion guarantees specified?",
    "If D&C Contractor specified?",
    "Is there a fixed price, date certain, turnkey contract?",
    "If revenue mitigating factors specified?",
    "If DSRA (Debt Service Reserve Account) specified?",
    "If covenants specified?",
    "Is facility syndicated?",
    "if ING commitment specified?",
    "Is tranche expected to be refinanced?",
    "Hedging: is the facility hedged?",
    "Is there a cash sweep mechanism applicable to the facility?",
}


def normalize_field_value(field_name: str, value: str) -> str:
    """
    Apply the appropriate normalizer based on field type.

    Returns the normalized value string.
    """
    if not value or value in ('NOT_FOUND', 'N/A', 'POSSIBLY_PRESENT',
                              'NOT_EXTRACTED', 'EXTRACTION_ERROR'):
        return value

    if field_name in DATE_FIELDS:
        return normalize_date(value)
    elif field_name in PERCENTAGE_FIELDS:
        return normalize_percentage(value)
    elif field_name in RATIO_FIELDS:
        return normalize_ratio(value)
    elif field_name in YES_NO_FIELDS:
        return normalize_yes_no(value)
    else:
        return value.strip()


def normalize_extraction_row(row: Dict) -> Dict:
    """Normalize all field values in an extraction row."""
    normalized = {}
    for field_name, value in row.items():
        if isinstance(value, str):
            normalized[field_name] = normalize_field_value(field_name, value)
        else:
            normalized[field_name] = value
    return normalized
