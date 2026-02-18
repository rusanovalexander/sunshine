"""
=================================================
Quality Comparison Tool
=================================================
Compares extraction results against a golden record
(e.g., Gemini 2.5 Pro output) to measure quality.

Usage:
    python -m src.quality_compare \
        --golden golden_record.csv \
        --extracted results_project_sunshine_v2.csv \
        --company "CompanyName"     # optional: filter to one company
        --output quality_report.json

The tool produces:
1. Per-field accuracy (exact match, fuzzy match, mismatch, both missing)
2. Per-company summary
3. Overall statistics
4. Detailed diff for every field mismatch
"""

import os
import re
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - QualityCompare - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# =====================================================================
# VALUE COMPARISON
# =====================================================================

SKIP_VALUES = {'NOT_FOUND', 'N/A', 'NOT_EXTRACTED', 'EXTRACTION_ERROR',
               'POSSIBLY_PRESENT', '', 'nan', 'None'}


def is_empty(value) -> bool:
    """Check if a value is effectively empty / not found."""
    if pd.isna(value):
        return True
    return str(value).strip() in SKIP_VALUES


def normalize_for_comparison(value) -> str:
    """Normalize a value for comparison (lowercase, strip whitespace/punctuation)."""
    if pd.isna(value):
        return ''
    s = str(value).strip().lower()
    # Remove trailing punctuation
    s = re.sub(r'[,;.\s]+$', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    return s


def fuzzy_number_match(a: str, b: str) -> bool:
    """Check if two strings represent the same number."""
    def extract_number(s):
        # Remove currency, commas, spaces
        cleaned = re.sub(r'[€$£¥,\s]', '', s)
        m = re.search(r'(\d+(?:\.\d+)?)', cleaned)
        return float(m.group(1)) if m else None

    num_a = extract_number(a)
    num_b = extract_number(b)

    if num_a is not None and num_b is not None:
        # Allow small floating-point tolerance
        if num_a == 0 and num_b == 0:
            return True
        if num_a == 0 or num_b == 0:
            return False
        return abs(num_a - num_b) / max(abs(num_a), abs(num_b)) < 0.01

    return False


def compare_values(golden_val, extracted_val) -> str:
    """
    Compare a golden record value against an extracted value.

    Returns one of:
    - "exact_match": Values are identical after normalization
    - "fuzzy_match": Values are semantically equivalent (same number, date, etc.)
    - "mismatch": Values differ
    - "both_missing": Both are empty/NOT_FOUND
    - "golden_only": Golden has value, extracted is missing
    - "extracted_only": Extracted has value, golden is missing
    """
    golden_empty = is_empty(golden_val)
    extracted_empty = is_empty(extracted_val)

    if golden_empty and extracted_empty:
        return "both_missing"
    if golden_empty and not extracted_empty:
        return "extracted_only"
    if not golden_empty and extracted_empty:
        return "golden_only"

    # Both have values — compare
    g_norm = normalize_for_comparison(golden_val)
    e_norm = normalize_for_comparison(extracted_val)

    if g_norm == e_norm:
        return "exact_match"

    # Check if one contains the other
    if g_norm in e_norm or e_norm in g_norm:
        return "fuzzy_match"

    # Numeric comparison
    if fuzzy_number_match(str(golden_val), str(extracted_val)):
        return "fuzzy_match"

    # Yes/No normalization
    yes_set = {'yes', 'true', 'y'}
    no_set = {'no', 'false', 'n'}
    if (g_norm in yes_set and e_norm in yes_set) or (g_norm in no_set and e_norm in no_set):
        return "exact_match"

    # Date format tolerance: "03/2024" vs "2024-03" vs "March 2024"
    g_date = _extract_month_year(g_norm)
    e_date = _extract_month_year(e_norm)
    if g_date and e_date and g_date == e_date:
        return "fuzzy_match"

    # Ratio tolerance: "1.20x" vs "1.20:1"
    g_ratio = _extract_ratio(g_norm)
    e_ratio = _extract_ratio(e_norm)
    if g_ratio is not None and e_ratio is not None and abs(g_ratio - e_ratio) < 0.01:
        return "fuzzy_match"

    return "mismatch"


def _extract_month_year(s: str) -> Optional[Tuple[int, int]]:
    """Try to extract (month, year) from a date string."""
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
        'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
    }

    # MM/YYYY
    m = re.match(r'(\d{1,2})/(\d{4})', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # YYYY-MM
    m = re.match(r'(\d{4})-(\d{1,2})', s)
    if m:
        return (int(m.group(2)), int(m.group(1)))

    # Month YYYY
    for name, num in month_map.items():
        m = re.search(rf'{name}\s+(\d{{4}})', s)
        if m:
            return (num, int(m.group(1)))

    return None


def _extract_ratio(s: str) -> Optional[float]:
    """Try to extract a ratio value from string."""
    # X.XXx
    m = re.match(r'(\d+(?:\.\d+)?)x', s)
    if m:
        return float(m.group(1))
    # X.XX:1
    m = re.match(r'(\d+(?:\.\d+)?)\s*:\s*1', s)
    if m:
        return float(m.group(1))
    return None


# =====================================================================
# COMPARISON ENGINE
# =====================================================================

@dataclass
class FieldComparison:
    field_name: str
    golden_value: str
    extracted_value: str
    result: str  # exact_match, fuzzy_match, mismatch, etc.


@dataclass
class CompanyComparison:
    company: str
    facility_index: int
    total_fields: int
    exact_matches: int
    fuzzy_matches: int
    mismatches: int
    golden_only: int
    extracted_only: int
    both_missing: int
    field_details: List[FieldComparison] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        matched = self.exact_matches + self.fuzzy_matches
        total_comparable = matched + self.mismatches + self.golden_only
        return matched / total_comparable if total_comparable > 0 else 0.0

    @property
    def coverage(self) -> float:
        """What % of golden record fields did we find (any value)?"""
        found = self.exact_matches + self.fuzzy_matches + self.mismatches + self.extracted_only
        total = self.total_fields - self.both_missing
        return found / total if total > 0 else 0.0


def compare_company(golden_row: pd.Series, extracted_row: pd.Series,
                    compare_fields: List[str], company: str,
                    facility_idx: int = 0) -> CompanyComparison:
    """Compare a single row (golden vs extracted) field by field."""
    results = {
        "exact_match": 0, "fuzzy_match": 0, "mismatch": 0,
        "golden_only": 0, "extracted_only": 0, "both_missing": 0,
    }
    details = []

    for field_name in compare_fields:
        g_val = golden_row.get(field_name, '')
        e_val = extracted_row.get(field_name, '')

        result = compare_values(g_val, e_val)
        results[result] += 1

        details.append(FieldComparison(
            field_name=field_name,
            golden_value=str(g_val) if not pd.isna(g_val) else '',
            extracted_value=str(e_val) if not pd.isna(e_val) else '',
            result=result,
        ))

    return CompanyComparison(
        company=company,
        facility_index=facility_idx,
        total_fields=len(compare_fields),
        exact_matches=results["exact_match"],
        fuzzy_matches=results["fuzzy_match"],
        mismatches=results["mismatch"],
        golden_only=results["golden_only"],
        extracted_only=results["extracted_only"],
        both_missing=results["both_missing"],
        field_details=details,
    )


# =====================================================================
# MAIN COMPARISON FLOW
# =====================================================================

def run_comparison(golden_csv: str, extracted_csv: str,
                   company_filter: str = None,
                   output_path: str = None) -> Dict:
    """
    Run quality comparison between golden record and extraction results.

    Args:
        golden_csv: Path to Gemini golden record CSV
        extracted_csv: Path to pipeline extraction CSV
        company_filter: Optional company name to filter
        output_path: Optional path to save JSON report

    Returns:
        Comparison report dict
    """
    logger.info(f"Loading golden record: {golden_csv}")
    golden_df = pd.read_csv(golden_csv, dtype=str)

    logger.info(f"Loading extraction results: {extracted_csv}")
    extracted_df = pd.read_csv(extracted_csv, dtype=str)

    # Find common columns (excluding metadata)
    metadata_cols = {'Client Folder', 'Source Files', 'Comments', 'Confidence Score'}
    golden_fields = set(golden_df.columns) - metadata_cols
    extracted_fields = set(extracted_df.columns) - metadata_cols
    common_fields = sorted(golden_fields & extracted_fields)

    logger.info(f"Common fields to compare: {len(common_fields)}")
    logger.info(f"Golden-only fields: {golden_fields - extracted_fields}")
    logger.info(f"Extracted-only fields: {extracted_fields - golden_fields}")

    # Determine company column
    company_col = 'Client Folder'
    if company_col not in golden_df.columns:
        # Try to find it
        for col in golden_df.columns:
            if 'company' in col.lower() or 'client' in col.lower() or 'folder' in col.lower():
                company_col = col
                break

    # Filter by company if requested
    if company_filter:
        golden_df = golden_df[golden_df[company_col].str.contains(company_filter, case=False, na=False)]
        extracted_df = extracted_df[extracted_df[company_col].str.contains(company_filter, case=False, na=False)]
        logger.info(f"Filtered to company '{company_filter}': "
                    f"{len(golden_df)} golden rows, {len(extracted_df)} extracted rows")

    if golden_df.empty:
        logger.error("No golden record rows found (after filtering)")
        return {}

    if extracted_df.empty:
        logger.error("No extraction rows found (after filtering)")
        return {}

    # Match rows by company name
    all_comparisons = []
    company_summaries = []

    golden_companies = golden_df[company_col].unique()

    for company_name in golden_companies:
        g_rows = golden_df[golden_df[company_col] == company_name]
        e_rows = extracted_df[extracted_df[company_col].str.contains(
            re.escape(str(company_name)), case=False, na=False
        )] if not pd.isna(company_name) else pd.DataFrame()

        if e_rows.empty:
            logger.warning(f"  No extraction found for: {company_name}")
            # Create a comparison showing all golden_only
            for g_idx, g_row in g_rows.iterrows():
                comp = CompanyComparison(
                    company=str(company_name),
                    facility_index=0,
                    total_fields=len(common_fields),
                    exact_matches=0, fuzzy_matches=0, mismatches=0,
                    golden_only=sum(1 for f in common_fields if not is_empty(g_row.get(f, ''))),
                    extracted_only=0,
                    both_missing=sum(1 for f in common_fields if is_empty(g_row.get(f, ''))),
                    field_details=[],
                )
                all_comparisons.append(comp)
            continue

        # Compare row-by-row (matching by facility index order)
        for fac_idx, (g_idx, g_row) in enumerate(g_rows.iterrows()):
            if fac_idx < len(e_rows):
                e_row = e_rows.iloc[fac_idx]
            else:
                # More golden rows than extracted — missing facility
                logger.warning(f"  Missing facility {fac_idx} for {company_name}")
                continue

            comp = compare_company(
                g_row, e_row, common_fields,
                company=str(company_name), facility_idx=fac_idx
            )
            all_comparisons.append(comp)

    # Aggregate statistics
    total_exact = sum(c.exact_matches for c in all_comparisons)
    total_fuzzy = sum(c.fuzzy_matches for c in all_comparisons)
    total_mismatch = sum(c.mismatches for c in all_comparisons)
    total_golden_only = sum(c.golden_only for c in all_comparisons)
    total_extracted_only = sum(c.extracted_only for c in all_comparisons)
    total_both_missing = sum(c.both_missing for c in all_comparisons)
    total_fields = sum(c.total_fields for c in all_comparisons)

    total_comparable = total_exact + total_fuzzy + total_mismatch + total_golden_only
    overall_accuracy = (total_exact + total_fuzzy) / total_comparable if total_comparable > 0 else 0.0

    # Per-field accuracy
    field_stats = {}
    for field_name in common_fields:
        f_results = {"exact_match": 0, "fuzzy_match": 0, "mismatch": 0,
                     "golden_only": 0, "extracted_only": 0, "both_missing": 0}
        for comp in all_comparisons:
            for detail in comp.field_details:
                if detail.field_name == field_name:
                    f_results[detail.result] += 1

        comparable = f_results["exact_match"] + f_results["fuzzy_match"] + \
                     f_results["mismatch"] + f_results["golden_only"]
        matched = f_results["exact_match"] + f_results["fuzzy_match"]
        accuracy = matched / comparable if comparable > 0 else None

        field_stats[field_name] = {
            **f_results,
            "accuracy": f"{accuracy:.1%}" if accuracy is not None else "N/A",
        }

    # Sort fields by accuracy (worst first for easy debugging)
    sorted_fields = sorted(
        field_stats.items(),
        key=lambda x: float(x[1]["accuracy"].replace('%', '')) / 100
        if x[1]["accuracy"] != "N/A" else 999
    )

    # Build report
    report = {
        "summary": {
            "golden_csv": golden_csv,
            "extracted_csv": extracted_csv,
            "company_filter": company_filter,
            "total_companies": len(all_comparisons),
            "total_fields_compared": total_fields,
            "overall_accuracy": f"{overall_accuracy:.1%}",
            "exact_matches": total_exact,
            "fuzzy_matches": total_fuzzy,
            "mismatches": total_mismatch,
            "golden_only_missing": total_golden_only,
            "extracted_only_extra": total_extracted_only,
            "both_missing": total_both_missing,
        },
        "per_field_accuracy": dict(sorted_fields),
        "per_company": [],
        "mismatches_detail": [],
    }

    # Per-company summaries
    for comp in all_comparisons:
        report["per_company"].append({
            "company": comp.company,
            "facility_index": comp.facility_index,
            "accuracy": f"{comp.accuracy:.1%}",
            "coverage": f"{comp.coverage:.1%}",
            "exact": comp.exact_matches,
            "fuzzy": comp.fuzzy_matches,
            "mismatch": comp.mismatches,
            "golden_only": comp.golden_only,
        })

        # Collect mismatches for debugging
        for detail in comp.field_details:
            if detail.result in ("mismatch", "golden_only"):
                report["mismatches_detail"].append({
                    "company": comp.company,
                    "facility": comp.facility_index,
                    "field": detail.field_name,
                    "golden": detail.golden_value,
                    "extracted": detail.extracted_value,
                    "result": detail.result,
                })

    # Print summary
    print("\n" + "=" * 70)
    print("QUALITY COMPARISON REPORT")
    print("=" * 70)
    print(f"Overall Accuracy: {report['summary']['overall_accuracy']}")
    print(f"  Exact matches:  {total_exact}")
    print(f"  Fuzzy matches:  {total_fuzzy}")
    print(f"  Mismatches:     {total_mismatch}")
    print(f"  Missing (ours): {total_golden_only}")
    print(f"  Extra (ours):   {total_extracted_only}")
    print(f"  Both missing:   {total_both_missing}")

    print(f"\nWorst fields (by accuracy):")
    for field_name, stats in sorted_fields[:15]:
        if stats["accuracy"] != "N/A":
            print(f"  {stats['accuracy']:>6s}  {field_name}")

    print(f"\nPer-company accuracy:")
    for comp_data in report["per_company"]:
        print(f"  {comp_data['accuracy']:>6s}  {comp_data['company']} "
              f"(facility {comp_data['facility_index']})")

    if report["mismatches_detail"]:
        print(f"\nTop mismatches ({min(20, len(report['mismatches_detail']))} shown):")
        for mm in report["mismatches_detail"][:20]:
            print(f"  [{mm['company']}] {mm['field']}")
            print(f"    Golden:    {mm['golden'][:80]}")
            print(f"    Extracted: {mm['extracted'][:80]}")

    print("=" * 70)

    # Save report
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Report saved to: {output_path}")

    return report


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare extraction results against a golden record"
    )
    parser.add_argument(
        "--golden", required=True,
        help="Path to golden record CSV (e.g., Gemini 2.5 Pro output)"
    )
    parser.add_argument(
        "--extracted", required=True,
        help="Path to extraction results CSV"
    )
    parser.add_argument(
        "--company",
        help="Filter comparison to a specific company"
    )
    parser.add_argument(
        "--output", default="quality_comparison.json",
        help="Output path for detailed JSON report"
    )

    args = parser.parse_args()

    if not os.path.exists(args.golden):
        logger.error(f"Golden record not found: {args.golden}")
        return 1

    if not os.path.exists(args.extracted):
        logger.error(f"Extraction results not found: {args.extracted}")
        return 1

    report = run_comparison(
        golden_csv=args.golden,
        extracted_csv=args.extracted,
        company_filter=args.company,
        output_path=args.output
    )

    if report:
        accuracy = report["summary"]["overall_accuracy"]
        logger.info(f"Comparison complete. Overall accuracy: {accuracy}")
        return 0
    return 1


if __name__ == "__main__":
    exit(main())
