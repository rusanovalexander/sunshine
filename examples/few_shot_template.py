"""
=================================================
Few-Shot Prompting Template
=================================================
Per-field-group examples for improved extraction accuracy.

Each field group gets its own targeted example that teaches
the LLM the exact output format, domain conventions, and
edge cases specific to that group.

Usage:
    from examples.few_shot_template import get_few_shot_for_group
    example_text = get_few_shot_for_group("pricing")
"""

import json
from typing import Dict, Optional


# =====================================================================
# PER-FIELD-GROUP FEW-SHOT EXAMPLES
# =====================================================================
# Each key matches a group name in config.FIELD_GROUPS.
# Examples are kept concise to minimize token overhead (~200-400 tokens each).

FIELD_GROUP_EXAMPLES: Dict[str, Dict] = {
    "basic_info": {
        "input_summary": (
            "SENIOR FACILITIES AGREEMENT dated 15 March 2024\n"
            "BORROWER: SolarCo Holdings B.V.\n"
            "FACILITY: EUR 250,000,000 Term Loan Facility\n"
            "Currency: Euro"
        ),
        "output": {
            "Borrower name": {
                "value": "SolarCo Holdings B.V.",
                "evidence": "BORROWER: SolarCo Holdings B.V.",
                "confidence": "HIGH"
            },
            "Senior facility name": {
                "value": "Senior Facilities Agreement",
                "evidence": "SENIOR FACILITIES AGREEMENT dated 15 March 2024",
                "confidence": "HIGH"
            },
            "Facility name": {
                "value": "Term Loan Facility",
                "evidence": "EUR 250,000,000 Term Loan Facility",
                "confidence": "HIGH"
            },
            "Type of tranche: Term Loan, Revolving Loan (RCF), Letter of Credit (LC), Bond": {
                "value": "Term Loan",
                "evidence": "EUR 250,000,000 Term Loan Facility",
                "confidence": "HIGH"
            },
            "Currency of the facility": {
                "value": "EUR",
                "evidence": "EUR 250,000,000 Term Loan Facility",
                "confidence": "HIGH"
            },
            "Credit limit ['000]": {
                "value": "EUR 250,000",
                "evidence": "EUR 250,000,000 Term Loan Facility",
                "confidence": "HIGH"
            },
        },
    },

    "sponsor_info": {
        "input_summary": (
            "The Project is owned by GreenEnergy SPV B.V. (the 'Borrower'), "
            "a wholly-owned subsidiary of GreenEnergy HoldCo B.V. (the 'HoldCo'). "
            "Meridian Capital Partners holds 75% of the equity in HoldCo."
        ),
        "output": {
            "Is a Sponsor linked to the project?": {
                "value": "Yes",
                "evidence": "Meridian Capital Partners holds 75% of the equity in HoldCo",
                "confidence": "HIGH"
            },
            "Sponsor Name": {
                "value": "Meridian Capital Partners",
                "evidence": "Meridian Capital Partners holds 75% of the equity",
                "confidence": "HIGH"
            },
            "Sponsor ownership [%]": {
                "value": "75%",
                "evidence": "holds 75% of the equity in HoldCo",
                "confidence": "HIGH"
            },
            "Is a HoldCo linked to the project?": {
                "value": "Yes",
                "evidence": "GreenEnergy HoldCo B.V. (the 'HoldCo')",
                "confidence": "HIGH"
            },
            "Name of the HoldCo linked to the project": {
                "value": "GreenEnergy HoldCo B.V.",
                "evidence": "GreenEnergy HoldCo B.V. (the 'HoldCo')",
                "confidence": "HIGH"
            },
        },
    },

    "project_details": {
        "input_summary": (
            "The Project comprises a 150MW onshore wind farm located in "
            "Andalusia, Spain. Construction is expected to complete in Q3 2025 "
            "with commercial operation date targeted for October 2025."
        ),
        "output": {
            "Project Sector (NACE Rev. 2 classification)": {
                "value": "D35.1.1 - Production of electricity",
                "evidence": "150MW onshore wind farm",
                "confidence": "MEDIUM"
            },
            "Project Sector (Aggregated)": {
                "value": "Wind and Solar",
                "evidence": "onshore wind farm",
                "confidence": "HIGH"
            },
            "Project location [Country]": {
                "value": "Spain",
                "evidence": "located in Andalusia, Spain",
                "confidence": "HIGH"
            },
            "Readiness [In operation / Construction needed]": {
                "value": "Construction needed",
                "evidence": "Construction is expected to complete in Q3 2025",
                "confidence": "HIGH"
            },
            "If construction needed: Readiness year": {
                "value": "2025",
                "evidence": "commercial operation date targeted for October 2025",
                "confidence": "HIGH"
            },
            "If construction needed: Readiness month": {
                "value": "10",
                "evidence": "targeted for October 2025",
                "confidence": "HIGH"
            },
        },
    },

    "construction_guarantees": {
        "input_summary": (
            "Vestas Wind Systems A/S has been appointed as the EPC contractor "
            "under a fixed-price, date-certain, turnkey contract. The Sponsor "
            "provides a completion guarantee until the COD."
        ),
        "output": {
            "If completion guarantees specified?": {
                "value": "Yes",
                "evidence": "The Sponsor provides a completion guarantee until the COD",
                "confidence": "HIGH"
            },
            "Guarantee Development sponsors name": {
                "value": "Meridian Capital Partners",
                "evidence": "The Sponsor provides a completion guarantee",
                "confidence": "MEDIUM"
            },
            "If D&C Contractor specified?": {
                "value": "Yes",
                "evidence": "Vestas Wind Systems A/S has been appointed as the EPC contractor",
                "confidence": "HIGH"
            },
            "Is there a fixed price, date certain, turnkey contract?": {
                "value": "Yes",
                "evidence": "fixed-price, date-certain, turnkey contract",
                "confidence": "HIGH"
            },
            "Name of the D&C Contractor": {
                "value": "Vestas Wind Systems A/S",
                "evidence": "Vestas Wind Systems A/S has been appointed as the EPC contractor",
                "confidence": "HIGH"
            },
        },
    },

    "covenants": {
        "input_summary": (
            "Financial covenants:\n"
            "- Historical DSCR: minimum 1.15:1 tested semi-annually\n"
            "- Projected DSCR: minimum 1.10x\n"
            "- LLCR: minimum 1.20:1\n"
            "- Net Debt / EBITDA: maximum 6.0x\n"
            "A DSRA of 6 months debt service is required.\n"
            "Dividend lock-up if B-DSCR < 1.20:1."
        ),
        "output": {
            "If DSRA (Debt Service Reserve Account) specified?": {
                "value": "Yes",
                "evidence": "A DSRA of 6 months debt service is required",
                "confidence": "HIGH"
            },
            "If covenants specified?": {
                "value": "Yes",
                "evidence": "Financial covenants: Historical DSCR: minimum 1.15:1",
                "confidence": "HIGH"
            },
            "Covenant leading to dividend lock-up: Backward-looking DSCR (B-DSCR)": {
                "value": "1.20x",
                "evidence": "Dividend lock-up if B-DSCR < 1.20:1",
                "confidence": "HIGH"
            },
            "Covenant leading to dividend lock-up: Forward-looking DSCR (F-DSCR)": {
                "value": "1.10x",
                "evidence": "Projected DSCR: minimum 1.10x",
                "confidence": "MEDIUM"
            },
            "Covenant leading to dividend lock-up: Loan Life Coverage Ratio (LLCR)": {
                "value": "1.20x",
                "evidence": "LLCR: minimum 1.20:1",
                "confidence": "HIGH"
            },
            "Covenant leading to dividend lock-up: Net Debt / EBITDA": {
                "value": "6.0x",
                "evidence": "Net Debt / EBITDA: maximum 6.0x",
                "confidence": "HIGH"
            },
        },
    },

    "syndication_ing": {
        "input_summary": (
            "The Facility is syndicated among: ING Bank N.V. (EUR 50,000,000), "
            "BNP Paribas (EUR 100,000,000), Rabobank (EUR 100,000,000). "
            "Total commitment: EUR 250,000,000."
        ),
        "output": {
            "Is facility syndicated?": {
                "value": "Yes",
                "evidence": "The Facility is syndicated among: ING Bank N.V., BNP Paribas, Rabobank",
                "confidence": "HIGH"
            },
            "if ING commitment specified?": {
                "value": "Yes",
                "evidence": "ING Bank N.V. (EUR 50,000,000)",
                "confidence": "HIGH"
            },
            "ING share [%]": {
                "value": "20%",
                "evidence": "ING Bank N.V. (EUR 50,000,000) / Total EUR 250,000,000 = 20%",
                "confidence": "HIGH"
            },
        },
    },

    "dates_schedules": {
        "input_summary": (
            "This Agreement is dated 15 March 2024. The Final Maturity Date "
            "is 15 March 2031 (7 years). Repayment: semi-annual instalments "
            "commencing 15 September 2025, each equal to 1/14th of the Loan."
        ),
        "output": {
            "Inception date [MM/YYYY] of the facility": {
                "value": "03/2024",
                "evidence": "This Agreement is dated 15 March 2024",
                "confidence": "HIGH"
            },
            "Maturity date [MM/YYYY] of the facility": {
                "value": "03/2031",
                "evidence": "The Final Maturity Date is 15 March 2031",
                "confidence": "HIGH"
            },
            "Is tranche expected to be refinanced?": {
                "value": "NOT_FOUND",
                "evidence": "",
                "confidence": "LOW"
            },
            "Principal repayment schedule": {
                "value": "Semi-annual equal instalments of 1/14th",
                "evidence": "semi-annual instalments commencing 15 September 2025, each equal to 1/14th",
                "confidence": "HIGH"
            },
        },
    },

    "pricing": {
        "input_summary": (
            "Interest Rate: EURIBOR (3-month) plus an Applicable Margin of "
            "2.75% per annum. EURIBOR floor of 0%. No interest rate cap applies."
        ),
        "output": {
            "Base Rate of the facility": {
                "value": "EURIBOR",
                "evidence": "Interest Rate: EURIBOR (3-month)",
                "confidence": "HIGH"
            },
            "Tenor of the base rate / Interest Rate Tenor": {
                "value": "3M",
                "evidence": "EURIBOR (3-month)",
                "confidence": "HIGH"
            },
            "Fix interest rate [%]": {
                "value": "NOT_FOUND",
                "evidence": "",
                "confidence": "LOW"
            },
            "Spread [%]": {
                "value": "2.75%",
                "evidence": "Applicable Margin of 2.75% per annum",
                "confidence": "HIGH"
            },
            "Interest rate floor %": {
                "value": "0%",
                "evidence": "EURIBOR floor of 0%",
                "confidence": "HIGH"
            },
            "Interest rate cap %": {
                "value": "NOT_FOUND",
                "evidence": "No interest rate cap applies",
                "confidence": "HIGH"
            },
        },
    },

    "hedging": {
        "input_summary": (
            "The Borrower has entered into an interest rate swap with ING Bank N.V. "
            "effective 15 June 2024, maturing 15 March 2031. Notional: EUR 200,000,000, "
            "fixed rate 3.25%, covering 80% of the floating rate exposure."
        ),
        "output": {
            "Hedging: is the facility hedged?": {
                "value": "Yes",
                "evidence": "The Borrower has entered into an interest rate swap",
                "confidence": "HIGH"
            },
            "Hedging: how is the facility hedged": {
                "value": "Interest rate swap",
                "evidence": "interest rate swap with ING Bank N.V.",
                "confidence": "HIGH"
            },
            "Hedging: effective date": {
                "value": "06/2024",
                "evidence": "effective 15 June 2024",
                "confidence": "HIGH"
            },
            "Hedging: fixed rate": {
                "value": "3.25%",
                "evidence": "fixed rate 3.25%",
                "confidence": "HIGH"
            },
            "Hedging: maturity date": {
                "value": "03/2031",
                "evidence": "maturing 15 March 2031",
                "confidence": "HIGH"
            },
            "Hedging: notional": {
                "value": "EUR 200,000,000",
                "evidence": "Notional: EUR 200,000,000",
                "confidence": "HIGH"
            },
            "Hedging: % of exposure hedged": {
                "value": "80%",
                "evidence": "covering 80% of the floating rate exposure",
                "confidence": "HIGH"
            },
        },
    },

    "cash_sweep": {
        "input_summary": (
            "A mandatory cash sweep mechanism applies: 50% of Excess Cash Flow "
            "shall be applied to prepay the Loan on each semi-annual payment date."
        ),
        "output": {
            "Is there a cash sweep mechanism applicable to the facility?": {
                "value": "Yes",
                "evidence": "A mandatory cash sweep mechanism applies",
                "confidence": "HIGH"
            },
            "Cash sweep structure": {
                "value": "Predetermined (% of cash)",
                "evidence": "50% of Excess Cash Flow shall be applied to prepay",
                "confidence": "HIGH"
            },
            "For predetermined - additional cash to be paid [%]": {
                "value": "50%",
                "evidence": "50% of Excess Cash Flow",
                "confidence": "HIGH"
            },
        },
    },

    "fees": {
        "input_summary": (
            "Fees:\n"
            "- Upfront Fee: 1.00% of the Total Commitments, payable on signing\n"
            "- Commitment Fee: 40% of the Applicable Margin on undrawn amounts\n"
            "- LC Fee: 1.50% per annum on outstanding LC amounts"
        ),
        "output": {
            "Upfront Fee [%]": {
                "value": "1.00%",
                "evidence": "Upfront Fee: 1.00% of the Total Commitments",
                "confidence": "HIGH"
            },
            "Commitment fee [%]": {
                "value": "40% of Applicable Margin",
                "evidence": "Commitment Fee: 40% of the Applicable Margin on undrawn amounts",
                "confidence": "HIGH"
            },
            "Letter of credit issuance fee [%]": {
                "value": "1.50%",
                "evidence": "LC Fee: 1.50% per annum on outstanding LC amounts",
                "confidence": "HIGH"
            },
        },
    },

    "revenue_mitigants": {
        "input_summary": (
            "The Project benefits from a 20-year Power Purchase Agreement with "
            "Endesa S.A. at a guaranteed price of EUR 85/MWh, covering 100% of "
            "the projected output."
        ),
        "output": {
            "If revenue mitigating factors specified?": {
                "value": "Yes",
                "evidence": "Power Purchase Agreement with Endesa S.A. at a guaranteed price",
                "confidence": "HIGH"
            },
            "Type of mitigating factor": {
                "value": "Guaranteed price",
                "evidence": "guaranteed price of EUR 85/MWh",
                "confidence": "HIGH"
            },
            "Contractual or regulatory factor guarantor": {
                "value": "Endesa S.A.",
                "evidence": "Power Purchase Agreement with Endesa S.A.",
                "confidence": "HIGH"
            },
            "Revenue mitigating factors applied to": {
                "value": "All revenues",
                "evidence": "covering 100% of the projected output",
                "confidence": "HIGH"
            },
            "Volume constraint type of revenue mitigating factors": {
                "value": "% of volume covered",
                "evidence": "covering 100% of the projected output",
                "confidence": "HIGH"
            },
            "Volume covered by revenue mitigating factors [%]": {
                "value": "100%",
                "evidence": "covering 100% of the projected output",
                "confidence": "HIGH"
            },
        },
    },
}


def format_few_shot_example(example: Dict) -> str:
    """Format an example for inclusion in prompts."""
    input_text = example["input_summary"].strip()

    output_lines = []
    for field, data in example["output"].items():
        output_lines.append(
            f'    "{field}": {{"value": "{data["value"]}", '
            f'"evidence": "{data["evidence"]}", '
            f'"confidence": "{data["confidence"]}"}}'
        )

    output_text = '  "fields": {\n' + ",\n".join(output_lines) + "\n  }"

    return (
        f"EXAMPLE INPUT:\n{input_text}\n\n"
        f"EXAMPLE OUTPUT:\n{{\n{output_text}\n}}"
    )


def get_few_shot_for_group(group_name: str) -> Optional[str]:
    """
    Get a formatted few-shot example for a specific field group.

    Returns None if no example exists for the group.
    Token cost: ~200-400 tokens per example.
    """
    example = FIELD_GROUP_EXAMPLES.get(group_name)
    if not example:
        return None
    return format_few_shot_example(example)


def create_enhanced_extraction_prompt(
    base_prompt: str,
    group_name: str = None,
) -> str:
    """
    Enhance extraction prompt with a field-group-specific few-shot example.

    Args:
        base_prompt: Original extraction prompt
        group_name: Field group name to get targeted example

    Returns:
        Enhanced prompt with example inserted before DOCUMENT TEXT
    """
    example_text = None

    if group_name:
        example_text = get_few_shot_for_group(group_name)

    if not example_text:
        return base_prompt

    # Insert example before the document text section
    if "DOCUMENT TEXT:" in base_prompt:
        parts = base_prompt.split("DOCUMENT TEXT:", 1)
        return (
            parts[0]
            + "\nHere is an example of a high-quality extraction:\n\n"
            + example_text
            + "\n\nNow extract from the following document with the same precision.\n\n"
            + "DOCUMENT TEXT:"
            + parts[1]
        )
    else:
        return base_prompt + "\n\n" + example_text
