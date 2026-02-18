"""
=================================================
Few-Shot Prompting Template
=================================================
Use this to inject high-quality extraction examples
into the LLM prompts for improved accuracy.

Usage:
    1. Add your example extractions to EXAMPLES list
    2. Import and use in extract_v2.py
"""

from typing import List, Dict

# Example extraction from a real document
# Replace with your own high-quality examples
EXAMPLES: List[Dict] = [
    {
        "input_summary": """
        SENIOR FACILITIES AGREEMENT dated 15 March 2024
        
        BORROWER: SolarCo Holdings B.V.
        FACILITY: EUR 250,000,000 Term Loan Facility
        MARGIN: 2.50% per annum over EURIBOR
        MATURITY: 7 years from signing
        
        The Borrower shall maintain a Debt Service Coverage Ratio 
        of not less than 1.20:1 tested semi-annually.
        
        ING Bank N.V. commitment: EUR 50,000,000 (20%)
        """,
        "output": {
            "Borrower name": {
                "value": "SolarCo Holdings B.V.",
                "evidence": "BORROWER: SolarCo Holdings B.V.",
                "confidence": "HIGH"
            },
            "Credit limit ['000]": {
                "value": "EUR 250,000,000",
                "evidence": "EUR 250,000,000 Term Loan Facility",
                "confidence": "HIGH"
            },
            "Spread [%]": {
                "value": "2.50%",
                "evidence": "MARGIN: 2.50% per annum over EURIBOR",
                "confidence": "HIGH"
            },
            "Base Rate of the facility": {
                "value": "EURIBOR",
                "evidence": "2.50% per annum over EURIBOR",
                "confidence": "HIGH"
            },
            "Maturity date [MM/YYYY] of the facility": {
                "value": "03/2031",
                "evidence": "7 years from signing (dated 15 March 2024)",
                "confidence": "MEDIUM"
            },
            "Covenant leading to dividend lock-up: Backward-looking DSCR (B-DSCR)": {
                "value": "1.20x",
                "evidence": "Debt Service Coverage Ratio of not less than 1.20:1",
                "confidence": "HIGH"
            },
            "ING share [%]": {
                "value": "20% (EUR 50M / EUR 250M)",
                "evidence": "ING Bank N.V. commitment: EUR 50,000,000 (20%)",
                "confidence": "HIGH"
            },
            "Is facility syndicated?": {
                "value": "Yes",
                "evidence": "Multiple lender commitments mentioned",
                "confidence": "MEDIUM"
            }
        }
    }
]


def format_few_shot_example(example: Dict) -> str:
    """Format an example for inclusion in prompts."""
    input_text = example["input_summary"].strip()
    
    output_lines = []
    for field, data in example["output"].items():
        output_lines.append(
            f'"{field}": {{"value": "{data["value"]}", '
            f'"evidence": "{data["evidence"]}", '
            f'"confidence": "{data["confidence"]}"}}'
        )
    
    output_text = "{\n  " + ",\n  ".join(output_lines) + "\n}"
    
    return f"""
=== EXAMPLE ===
INPUT:
{input_text}

OUTPUT:
{output_text}
=== END EXAMPLE ===
"""


def get_few_shot_prompt_addition() -> str:
    """Get formatted few-shot examples for prompt injection."""
    if not EXAMPLES:
        return ""
    
    examples_text = "\n".join(format_few_shot_example(ex) for ex in EXAMPLES)
    
    return f"""
Here are examples of high-quality extractions:
{examples_text}

Now extract from the following document using the same format and attention to detail:
"""


def create_enhanced_extraction_prompt(
    base_prompt: str, 
    use_few_shot: bool = True
) -> str:
    """
    Enhance extraction prompt with few-shot examples.
    
    Args:
        base_prompt: Original extraction prompt
        use_few_shot: Whether to include examples
    
    Returns:
        Enhanced prompt with examples
    """
    if not use_few_shot:
        return base_prompt
    
    few_shot_addition = get_few_shot_prompt_addition()
    
    # Insert examples before the document text
    if "DOCUMENT TEXT:" in base_prompt:
        parts = base_prompt.split("DOCUMENT TEXT:", 1)
        return parts[0] + few_shot_addition + "\nDOCUMENT TEXT:" + parts[1]
    else:
        return base_prompt + "\n" + few_shot_addition


# =============================================================================
# INTEGRATION GUIDE
# =============================================================================
"""
To use few-shot prompting in extract_v2.py:

1. Import this module:
   from examples.few_shot_template import create_enhanced_extraction_prompt

2. In extract_field_group(), enhance the prompt:
   user_prompt = EXTRACTION_USER_TEMPLATE.format(...)
   user_prompt = create_enhanced_extraction_prompt(user_prompt)

3. Add your own examples to the EXAMPLES list above

Benefits:
- 60-80% of fine-tuning improvement without training
- Easy to update examples as you find better ones
- Works with any LLM model
"""
