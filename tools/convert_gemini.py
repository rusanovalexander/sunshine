"""
=================================================
Gemini XLSX → Pipeline CSV Converter
=================================================
Converts a Gemini 2.5 Pro extraction result (XLSX)
into the pipeline's CSV format so that quality_compare.py
can be used for comparison.

Usage:
    python -m tools.convert_gemini \
        --input  gemini_etalon.xlsx \
        --output golden_record.csv \
        --sheet  "Sheet1"               # optional, default first sheet

Then compare with:
    python -m src.quality_compare \
        --golden golden_record.csv \
        --extracted results_project_sunshine_v2.csv
"""

import argparse
import logging
import sys
import os

import pandas as pd

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ALL_FIELDS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - GeminiConverter - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# =====================================================================
# COLUMN MAPPING: Gemini column name → Pipeline column name
# =====================================================================
# Gemini uses longer/different names for some fields. This mapping
# converts Gemini column names to the exact ALL_FIELDS names used
# by the pipeline and quality_compare.py.
#
# Columns with identical names in both systems are handled automatically
# and do NOT need to be listed here.

GEMINI_TO_PIPELINE = {
    # Sponsor
    "Sponsor ownership (%)":
        "Sponsor ownership [%]",

    # HoldCo
    "Is a holding company linked to the project (i.e., if revenues of the project will also partially flow to a HoldCo)?":
        "Is a HoldCo linked to the project?",
    "Name of the holding company linked to the project":
        "Name of the HoldCo linked to the project",

    # Construction
    "If Design and Construct Contractor specified?":
        "If D&C Contractor specified?",
    "Is there a fixed price, date certain, turnkey contract (meaning that all cost overruns are borne by the D&C Contractor)?":
        "Is there a fixed price, date certain, turnkey contract?",
    "Name of the Design and Construct Contractor (or if Design and Construct Contractor benefits from a Parent Company Guarantee, please provide name of the Guarantor)":
        "Name of the D&C Contractor",

    # Revenue mitigants
    "Type of mitigating factor [Guaranteed price (not in '000) / Guaranteed add-on to mkt price (not in '000) / Fix revenue component (in '000) / Regulated rate-of-return (not in '000) / Variable OPEX passed through to off-taker (%)]":
        "Type of mitigating factor",
    "Revenue mitigating factors applied to [all revenues / revenues of the borrower]":
        "Revenue mitigating factors applied to",
    "Volume constraint type of revenue mitigating factors [Absolute volume / % of volume covered]":
        "Volume constraint type of revenue mitigating factors",

    # DSRA
    "DSRA (Debt Service Reserve Account)":
        "If DSRA (Debt Service Reserve Account) specified?",

    # Covenants
    "if covenants specified?":
        "If covenants specified?",
    "Covenant leading to dividend lock-up: Backward-looking Debt Service Coverage Ratio (B-DSCR)":
        "Covenant leading to dividend lock-up: Backward-looking DSCR (B-DSCR)",
    "Covenant leading to dividend lock-up: Forward-looking Debt Service Coverage Ratio (F-DSCR)":
        "Covenant leading to dividend lock-up: Forward-looking DSCR (F-DSCR)",

    # Credit limit
    "Credit limit [ '000] (in accounting format with currency)":
        "Credit limit ['000]",

    # ING
    "ING share [%] (if no % provided, divide commitment of ING by total commitment + provide applied formula)":
        "ING share [%]",

    # Tranche
    "Is tranche expected to be refinanced?":
        "Is tranche expected to be refinanced?",

    # Schedules
    "Principal drawdown schedule (%) (if no % is provided, divide the drawdown amount by limit amount)":
        "Principal drawdown schedule [Date: %]",
    "Principal repayment schedule (%) (if no % is provided, divide the repayment amount by limit amount)":
        "Principal repayment schedule [Date: %]",

    # Pricing
    "Fix interest rate [%] (if no base rate is selected)":
        "Fix interest rate [%]",
    "Spread [%] (if a base rate is agreed)":
        "Spread [%]",
    "Interest rate floor % (if applicable)":
        "Interest rate floor %",
    "Interest rate cap % (if applicable)":
        "Interest rate cap %",

    # Hedging
    "Hedging: is the facility hedged":
        "Hedging: is the facility hedged?",
    "Hedging: how is the facility hedged (e.g. using interest rate, FX swaps, inflation swaps)":
        "Hedging: how is the facility hedged",
    "Hedging: effective date":
        "Hedging: effective date",
    "Hedging: fixed rate":
        "Hedging: fixed rate",
    "Hedging: maturity date":
        "Hedging: maturity date",
    "Hedging: notional":
        "Hedging: notional",
    "Hedging: spread":
        "Hedging: spread",
    "% of exposure hedged":
        "Hedging: % of exposure hedged",

    # Cash sweep
    "For predetermined - additional cash to be paid [% of additional cash available after debt service]":
        "For predetermined - additional cash to be paid [%]",
    "For targeted - target repayment profile [% of base repayment]":
        "For targeted - target repayment profile [%]",
    "For transmission asset linked loan - sweep percentage [% of cash available]":
        "For transmission asset linked loan - sweep percentage [%]",

    # Fees
    "Upfront Fee %":
        "Upfront Fee [%]",
    "Commitment fee %":
        "Commitment fee [%]",
    "Letter of credit issuance fee %":
        "Letter of credit issuance fee [%]",
}

# Build reverse mapping for diagnostics
_PIPELINE_TO_GEMINI = {v: k for k, v in GEMINI_TO_PIPELINE.items()}


def convert_gemini_to_pipeline(input_path: str, output_path: str,
                               sheet_name=0) -> pd.DataFrame:
    """
    Read a Gemini XLSX file and convert it to the pipeline CSV format.

    Args:
        input_path: Path to Gemini XLSX file
        output_path: Path to write the output CSV
        sheet_name: Sheet name or index (default: first sheet)

    Returns:
        The converted DataFrame
    """
    logger.info(f"Reading Gemini file: {input_path}")

    if input_path.endswith('.xlsx') or input_path.endswith('.xls'):
        df = pd.read_excel(input_path, sheet_name=sheet_name, dtype=str)
    elif input_path.endswith('.csv'):
        df = pd.read_csv(input_path, dtype=str)
    elif input_path.endswith('.tsv') or input_path.endswith('.txt'):
        df = pd.read_csv(input_path, sep='\t', dtype=str)
    else:
        # Try xlsx first, fall back to csv
        try:
            df = pd.read_excel(input_path, sheet_name=sheet_name, dtype=str)
        except Exception:
            df = pd.read_csv(input_path, dtype=str)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # --- Step 1: Rename columns using the mapping ---
    renamed = {}
    unmapped_gemini = []

    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in GEMINI_TO_PIPELINE:
            renamed[col] = GEMINI_TO_PIPELINE[col_stripped]
        elif col_stripped in ALL_FIELDS:
            # Already matches pipeline name exactly
            renamed[col] = col_stripped
        else:
            # Try case-insensitive match
            matched = False
            for pipeline_field in ALL_FIELDS:
                if col_stripped.lower() == pipeline_field.lower():
                    renamed[col] = pipeline_field
                    matched = True
                    break
            if not matched:
                unmapped_gemini.append(col_stripped)
                renamed[col] = col_stripped  # Keep as-is

    df = df.rename(columns=renamed)

    # --- Step 2: Report mapping results ---
    pipeline_set = set(ALL_FIELDS)
    converted_cols = set(df.columns)

    mapped_fields = sorted(pipeline_set & converted_cols)
    missing_from_gemini = sorted(pipeline_set - converted_cols)
    extra_in_gemini = sorted(converted_cols - pipeline_set)

    logger.info(f"\nColumn mapping results:")
    logger.info(f"  Mapped to pipeline fields: {len(mapped_fields)}")
    logger.info(f"  Pipeline fields missing from Gemini: {len(missing_from_gemini)}")
    if missing_from_gemini:
        for f in missing_from_gemini:
            logger.info(f"    - {f}")
    logger.info(f"  Extra Gemini columns (no pipeline match): {len(extra_in_gemini)}")
    if extra_in_gemini:
        for f in extra_in_gemini:
            logger.info(f"    - {f}")

    if unmapped_gemini:
        logger.warning(f"\n  Unmapped Gemini columns (kept as-is):")
        for f in unmapped_gemini:
            logger.warning(f"    - {f}")

    # --- Step 3: Reorder columns to match ALL_FIELDS ---
    ordered_cols = [c for c in ALL_FIELDS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra_cols]

    # --- Step 4: Add missing pipeline columns with empty values ---
    for field in ALL_FIELDS:
        if field not in df.columns:
            df[field] = ""

    # Re-order to exact ALL_FIELDS order
    df = df[[c for c in ALL_FIELDS] + [c for c in df.columns if c not in ALL_FIELDS]]

    # --- Step 5: Clean values ---
    df = df.fillna("")

    # --- Step 6: Save ---
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"\nSaved {len(df)} rows to {output_path}")
    logger.info(f"Columns: {len([c for c in df.columns if c in pipeline_set])} pipeline fields "
                f"+ {len(extra_cols)} extra")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert Gemini XLSX extraction to pipeline CSV format"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to Gemini extraction file (XLSX, CSV, or TSV)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to output CSV file (pipeline format)"
    )
    parser.add_argument(
        "--sheet", "-s", default=0,
        help="Sheet name or index for XLSX files (default: first sheet)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1

    # Try to parse sheet as integer
    sheet = args.sheet
    try:
        sheet = int(sheet)
    except (ValueError, TypeError):
        pass

    convert_gemini_to_pipeline(args.input, args.output, sheet_name=sheet)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
