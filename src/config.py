"""
=================================================
Configuration for Document Extraction Pipeline v2
=================================================
Central configuration with field groups, prompts, and settings.
"""

import os

# =====================================================================
# PATHS
# =====================================================================
MODEL_PATH = "/home/inghero/data/irwbds/llm/parc/qwen3_14B"
VLM_MODEL_PATH = "/home/inghero/data/irwbds/llm/parc/Qwen2.5-VL-3B-Instruct"
EMBEDDING_MODEL_PATH = "/home/inghero/data/irwbds/llm/parc/Qwen3-Embedding-0.6B"

ARCHIVE_PATH = "/home/inghero/data/irwbds/llm/parc/notebooks/PF_Case/pipline_pf_2/clients_pf_2.zip"
EXTRACTED_SOURCE_DIR = "extracted_source_data_v2"
PREPROCESSED_DATA_DIR = "preprocessed_data_v2"
CHUNKS_DIR = "chunks_v2"
EXTRACTION_DIR = "extraction_outputs_v2"
OUTPUT_CSV = "results_project_sunshine_v2.csv"

# =====================================================================
# PROCESSING SETTINGS
# =====================================================================
CHUNK_SIZE = 2000  # tokens per chunk
CHUNK_OVERLAP = 400  # overlap between chunks
MAX_CHUNKS_PER_FIELD_GROUP = 5  # max chunks to send to LLM per extraction
                                # 5 × 3000 = ~15K tokens — fits comfortably in 20GB MIG
                                # (was 12 = ~36K tokens → KV cache OOM)
DPI = 300  # for PDF rendering

# Retriever Settings
RETRIEVER_TYPE = "bm25"  # Options: "bm25", "embedding", "hybrid"
EMBEDDING_BATCH_SIZE = 32  # Batch size for encoding chunks during indexing
HYBRID_BM25_WEIGHT = 0.5  # Weight for BM25 in hybrid mode (embedding gets 1 - this)

# LLM Settings
MAX_NEW_TOKENS = 2048  # Reduced from 4096 to fit in 20GB MIG KV cache budget
TEMPERATURE = 0.1
TOP_P = 0.95
REPETITION_PENALTY = 1.1

# =====================================================================
# FIELD GROUPS - Logical groupings for multi-pass extraction
# =====================================================================
FIELD_GROUPS = {
    "basic_info": {
        "name": "Basic Facility Information",
        "fields": [
            ("Borrower name", "The legal name of the borrowing entity/company"),
            ("Senior facility name", "Name of the senior credit facility"),
            ("Facility name", "Specific name of this facility/tranche"),
            ("Type of tranche: Term Loan, Revolving Loan (RCF), Letter of Credit (LC), Bond", 
             "The type of credit facility"),
            ("Currency of the facility", "Currency denomination (EUR, USD, GBP, etc.)"),
            ("Credit limit ['000]", "Total credit limit amount with currency"),
        ],
        "keywords": ["borrower", "facility", "agreement", "tranche", "loan", "credit", 
                     "limit", "commitment", "currency", "principal"]
    },
    
    "sponsor_info": {
        "name": "Sponsor and Ownership",
        "fields": [
            ("Is a Sponsor linked to the project?", "Yes/No - whether there's a project sponsor"),
            ("Sponsor Name", "Name of the sponsor entity"),
            ("Sponsor ownership [%]", "Percentage ownership by sponsor"),
            ("Is a HoldCo linked to the project?", "Yes/No - whether a holding company is involved"),
            ("Name of the HoldCo linked to the project", "Name of the holding company"),
        ],
        "keywords": ["sponsor", "ownership", "shareholder", "equity", "holding", "parent",
                     "investor", "stake", "percentage", "owned"]
    },
    
    "project_details": {
        "name": "Project Details",
        "fields": [
            ("Project Sector (NACE Rev. 2 classification)", "NACE sector code and description"),
            ("Project Sector (Aggregated)", 
             "Must be ONE of: Energy / Infrastructure / LNG / Manufacturing / Mining / TM / Wind and Solar"),
            ("Project location [Country]", "Country where the project is located"),
            ("Readiness [In operation / Construction needed]", "Current project status"),
            ("If construction needed: Readiness year", "Expected completion year"),
            ("If construction needed: Readiness month", "Expected completion month"),
        ],
        "keywords": ["project", "sector", "location", "country", "construction", "operation",
                     "completion", "commercial", "NACE", "industry", "site"]
    },
    
    "construction_guarantees": {
        "name": "Construction and Guarantees",
        "fields": [
            ("If completion guarantees specified?", "Yes/No - presence of completion guarantees"),
            ("Guarantee Development sponsors name", "Name of guarantee provider"),
            ("If D&C Contractor specified?", "Yes/No - presence of D&C contractor"),
            ("Is there a fixed price, date certain, turnkey contract?", 
             "Yes/No - whether cost overruns are borne by contractor"),
            ("Name of the D&C Contractor", "Name of Design & Construction contractor or guarantor"),
        ],
        "keywords": ["completion", "guarantee", "contractor", "EPC", "turnkey", "construction",
                     "D&C", "fixed price", "cost overrun", "performance"]
    },
    
    "revenue_mitigants": {
        "name": "Revenue Mitigating Factors",
        "fields": [
            ("If revenue mitigating factors specified?", "Yes/No"),
            ("Type of mitigating factor", 
             "Guaranteed price / Guaranteed add-on / Fix revenue component / Regulated rate-of-return / Variable OPEX passed through"),
            ("Contractual or regulatory factor guarantor", "Entity providing the guarantee"),
            ("Revenue mitigating factors applied to", "all revenues / revenues of the borrower"),
            ("Volume constraint type of revenue mitigating factors", "Absolute volume / % of volume covered"),
            ("Volume covered by revenue mitigating factors [%]", "Percentage of volume covered"),
        ],
        "keywords": ["revenue", "offtake", "PPA", "power purchase", "guaranteed", "tariff",
                     "regulated", "price", "volume", "contract", "off-taker"]
    },
    
    "covenants": {
        "name": "Financial Covenants",
        "fields": [
            ("If DSRA (Debt Service Reserve Account) specified?", "Yes/No"),
            ("If covenants specified?", "Yes/No"),
            ("Covenant leading to dividend lock-up: Minimum cash, '000", "Minimum cash covenant"),
            ("Covenant leading to dividend lock-up: Minimum Equity Reserve Account, '000", "Min equity reserve"),
            ("Covenant leading to dividend lock-up: Backward-looking DSCR (B-DSCR)", "Historical DSCR ratio"),
            ("Covenant leading to dividend lock-up: Forward-looking DSCR (F-DSCR)", "Projected DSCR ratio"),
            ("Covenant leading to dividend lock-up: Loan Life Coverage Ratio (LLCR)", "LLCR covenant"),
            ("Covenant leading to dividend lock-up: Interest Coverage Ratio (ICR)", "ICR covenant"),
            ("Covenant leading to dividend lock-up: Net Debt / EBITDA", "Leverage ratio"),
            ("Covenant leading to dividend lock-up: Dividend Cover Ratio", "Dividend coverage"),
            ("Covenant leading to dividend lock-up: Net Debt / RAB", "Regulatory asset base ratio"),
            ("Covenant leading to dividend lock-up: Debt / Equity", "D/E ratio"),
        ],
        "keywords": ["covenant", "DSCR", "LLCR", "ICR", "EBITDA", "coverage", "ratio",
                     "lock-up", "dividend", "reserve", "debt service", "minimum"]
    },
    
    "syndication_ing": {
        "name": "Syndication and ING Participation",
        "fields": [
            ("Is facility syndicated?", "Yes/No"),
            ("if ING commitment specified?", "Yes/No"),
            ("ING share [%]", "ING's percentage share - calculate if needed: (ING Amount / Total Amount)"),
        ],
        "keywords": ["syndicate", "syndication", "ING", "participant", "lender", "bank",
                     "commitment", "share", "proportion", "underwrite"]
    },
    
    "dates_schedules": {
        "name": "Dates and Schedules",
        "fields": [
            ("Inception date [MM/YYYY] of the facility", "Start date of facility"),
            ("Maturity date [MM/YYYY] of the facility", "End date of facility"),
            ("Is tranche expected to be refinanced?", "Yes/No"),
            ("Principal drawdown schedule", "Description of drawdown schedule"),
            ("Principal drawdown schedule [Date: %]", "Specific dates and percentages for drawdowns"),
            ("Principal repayment schedule", "Description of repayment schedule"),
            ("Principal repayment schedule [Date: %]", "Specific dates and percentages for repayments"),
        ],
        "keywords": ["inception", "maturity", "schedule", "drawdown", "repayment", "amortization",
                     "date", "term", "tenor", "bullet", "refinance"]
    },
    
    "pricing": {
        "name": "Interest Rates and Pricing",
        "fields": [
            ("Base Rate of the facility", "Reference rate (EURIBOR, SOFR, LIBOR, etc.)"),
            ("Tenor of the base rate / Interest Rate Tenor", "Period for interest rate (1M, 3M, 6M, etc.)"),
            ("Fix interest rate [%]", "Fixed rate if no base rate"),
            ("Spread [%]", "Margin over base rate"),
            ("Interest rate floor %", "Minimum interest rate"),
            ("Interest rate cap %", "Maximum interest rate"),
        ],
        "keywords": ["interest", "rate", "spread", "margin", "EURIBOR", "SOFR", "LIBOR",
                     "basis points", "bps", "floor", "cap", "pricing"]
    },
    
    "hedging": {
        "name": "Hedging Arrangements",
        "fields": [
            ("Hedging: is the facility hedged?", "Yes/No"),
            ("Hedging: how is the facility hedged", "interest rate swaps / FX swaps / inflation swaps"),
            ("Hedging: effective date", "Start date of hedge"),
            ("Hedging: fixed rate", "Fixed rate in swap"),
            ("Hedging: maturity date", "End date of hedge"),
            ("Hedging: notional", "Notional amount of hedge"),
            ("Hedging: spread", "Spread in hedge"),
            ("Hedging: % of exposure hedged", "Percentage of exposure covered"),
        ],
        "keywords": ["hedge", "hedging", "swap", "derivative", "interest rate swap", "IRS",
                     "FX", "currency", "notional", "fixed", "floating"]
    },
    
    "cash_sweep": {
        "name": "Cash Sweep Mechanism",
        "fields": [
            ("Is there a cash sweep mechanism applicable to the facility?", "Yes/No"),
            ("Cash sweep structure", 
             "Predetermined (% of cash) / Targeted debt balance / Transmission asset linked loan"),
            ("For predetermined - additional cash to be paid [%]", "Percentage of excess cash"),
            ("For targeted - target repayment profile [%]", "Target as % of base repayment"),
            ("For transmission asset linked loan - sweep percentage [%]", "Sweep percentage"),
        ],
        "keywords": ["cash sweep", "sweep", "excess cash", "mandatory prepayment", 
                     "cash waterfall", "surplus"]
    },
    
    "fees": {
        "name": "Fees",
        "fields": [
            ("Upfront Fee [%]", "Initial fee percentage"),
            ("Commitment fee [%]", "Fee on undrawn amounts"),
            ("Letter of credit issuance fee [%]", "LC fee"),
        ],
        "keywords": ["fee", "upfront", "commitment", "arrangement", "agency", "LC fee",
                     "letter of credit", "issuance"]
    },
}

# =====================================================================
# ALL FIELDS IN ORDER (for final CSV)
# =====================================================================
ALL_FIELDS = [
    "Client Folder", "Source Files", "Comments", "Confidence Score",
    "Borrower name", "Senior facility name", "Facility name",
    "Is a Sponsor linked to the project?", "Sponsor Name", "Sponsor ownership [%]",
    "Project Sector (NACE Rev. 2 classification)",
    "Project Sector (Aggregated)",
    "Project location [Country]",
    "Is a HoldCo linked to the project?",
    "Name of the HoldCo linked to the project",
    "Readiness [In operation / Construction needed]",
    "If construction needed: Readiness year",
    "If construction needed: Readiness month",
    "If completion guarantees specified?",
    "Guarantee Development sponsors name",
    "If D&C Contractor specified?",
    "Is there a fixed price, date certain, turnkey contract?",
    "Name of the D&C Contractor",
    "If revenue mitigating factors specified?",
    "Type of mitigating factor",
    "Contractual or regulatory factor guarantor",
    "Revenue mitigating factors applied to",
    "Volume constraint type of revenue mitigating factors",
    "Volume covered by revenue mitigating factors [%]",
    "If DSRA (Debt Service Reserve Account) specified?",
    "If covenants specified?",
    "Covenant leading to dividend lock-up: Minimum cash, '000",
    "Covenant leading to dividend lock-up: Minimum Equity Reserve Account, '000",
    "Covenant leading to dividend lock-up: Backward-looking DSCR (B-DSCR)",
    "Covenant leading to dividend lock-up: Forward-looking DSCR (F-DSCR)",
    "Covenant leading to dividend lock-up: Loan Life Coverage Ratio (LLCR)",
    "Covenant leading to dividend lock-up: Interest Coverage Ratio (ICR)",
    "Covenant leading to dividend lock-up: Net Debt / EBITDA",
    "Covenant leading to dividend lock-up: Dividend Cover Ratio",
    "Covenant leading to dividend lock-up: Net Debt / RAB",
    "Covenant leading to dividend lock-up: Debt / Equity",
    "Credit limit ['000]",
    "Is facility syndicated?",
    "if ING commitment specified?",
    "ING share [%]",
    "Type of tranche: Term Loan, Revolving Loan (RCF), Letter of Credit (LC), Bond",
    "Is tranche expected to be refinanced?",
    "Currency of the facility",
    "Inception date [MM/YYYY] of the facility",
    "Maturity date [MM/YYYY] of the facility",
    "Principal drawdown schedule",
    "Principal drawdown schedule [Date: %]",
    "Principal repayment schedule",
    "Principal repayment schedule [Date: %]",
    "Base Rate of the facility",
    "Tenor of the base rate / Interest Rate Tenor",
    "Fix interest rate [%]",
    "Spread [%]",
    "Interest rate floor %",
    "Interest rate cap %",
    "Hedging: is the facility hedged?",
    "Hedging: how is the facility hedged",
    "Hedging: effective date",
    "Hedging: fixed rate",
    "Hedging: maturity date",
    "Hedging: notional",
    "Hedging: spread",
    "Hedging: % of exposure hedged",
    "Is there a cash sweep mechanism applicable to the facility?",
    "Cash sweep structure",
    "For predetermined - additional cash to be paid [%]",
    "For targeted - target repayment profile [%]",
    "For transmission asset linked loan - sweep percentage [%]",
    "Upfront Fee [%]",
    "Commitment fee [%]",
    "Letter of credit issuance fee [%]",
]

# Fields that require extraction (excluding metadata)
EXTRACTABLE_FIELDS = [f for f in ALL_FIELDS if f not in
                      ["Client Folder", "Source Files", "Comments", "Confidence Score"]]

# =====================================================================
# ADAPTIVE MAX_NEW_TOKENS PER FIELD GROUP
# =====================================================================
# Groups with fewer/simpler fields need fewer output tokens.
# This reduces peak KV cache memory and speeds up inference on A100 20GB.
FIELD_GROUP_MAX_TOKENS = {
    "basic_info": 2000,
    "sponsor_info": 1500,
    "project_details": 1500,
    "construction_guarantees": 1500,
    "revenue_mitigants": 2000,
    "covenants": 3500,       # 12 fields, many sub-values
    "syndication_ing": 800,  # 3 simple fields
    "dates_schedules": 2500, # schedules can be verbose
    "pricing": 1500,
    "hedging": 2500,         # 8 fields
    "cash_sweep": 1200,
    "fees": 800,             # 3 simple fields
}

# =====================================================================
# FIELD-GROUP-SPECIFIC SYSTEM PROMPTS
# =====================================================================
# Each group gets a specialized system prompt that teaches the LLM
# domain-specific conventions. Same token budget as before (replacing
# the generic prompt, not adding to it).

FIELD_GROUP_SYSTEM_PROMPTS = {
    "covenants": """You are an expert financial analyst specializing in project finance covenants.
Your task is to extract financial covenant data with extreme precision.

DOMAIN RULES:
- Ratios like 1.20:1 mean 1.20x. Always output as X.XXx format.
- DSCR = Debt Service Coverage Ratio. Look for "debt service coverage", "DSCR", "DS coverage".
- LLCR = Loan Life Coverage Ratio. PLCR = Project Life Coverage Ratio.
- ICR = Interest Coverage Ratio. May appear as "interest cover".
- "Lock-up" / "distribution stopper" / "dividend trap" all mean dividend restriction.
- Backward-looking DSCR is historical. Forward-looking DSCR is projected.
- DSRA = Debt Service Reserve Account. Check for months of debt service required.
- Net Debt / EBITDA is a leverage ratio — look for "gearing", "leverage covenant".

CRITICAL RULES:
1. ONLY extract information EXPLICITLY stated in the document
2. Provide exact quotes as evidence
3. Use "NOT_FOUND" if clearly absent, "POSSIBLY_PRESENT" if likely in another section
4. Output valid JSON.""",

    "pricing": """You are an expert financial analyst specializing in loan pricing and interest rates.

DOMAIN RULES:
- Convert basis points to percentages: 250bps = 2.50%
- "Applicable Margin" / "Credit Spread" / "Margin" all mean the Spread
- Base rate = reference rate (EURIBOR, SOFR, LIBOR, SONIA, etc.)
- Tenor means the interest period: 1M, 3M, 6M, 12M
- Floor = minimum rate. Cap = maximum rate.
- If rate is fixed (no base rate), extract as Fix interest rate
- If rate is floating (base + spread), extract base rate AND spread separately

CRITICAL RULES:
1. ONLY extract information EXPLICITLY stated in the document
2. Provide exact quotes as evidence
3. Use "NOT_FOUND" if clearly absent, "POSSIBLY_PRESENT" if likely in another section
4. Output valid JSON.""",

    "dates_schedules": """You are an expert financial analyst extracting dates and payment schedules.

DOMAIN RULES:
- Convert ALL dates to MM/YYYY format
- "7 years from signing date of March 2024" → maturity = 03/2031
- Inception = signing date / effective date / closing date / commencement date
- Maturity = final repayment date / termination date / expiry date
- "Bullet" repayment = 100% at maturity (no amortization)
- "Semi-annual" = every 6 months. "Quarterly" = every 3 months.
- Drawdown schedule: when money is disbursed. Repayment schedule: when it's paid back.

CRITICAL RULES:
1. ONLY extract information EXPLICITLY stated in the document
2. Provide exact quotes as evidence
3. Use "NOT_FOUND" if clearly absent, "POSSIBLY_PRESENT" if likely in another section
4. Output valid JSON.""",

    "hedging": """You are an expert financial analyst specializing in hedging and derivatives.

DOMAIN RULES:
- Interest Rate Swap (IRS): borrower pays fixed, receives floating
- FX Swap: hedges currency exposure
- "Notional" / "Nominal" = the reference amount the swap is based on
- Fixed rate in a swap is the rate the borrower pays
- "% of exposure hedged" = notional / facility amount
- Effective date = when the swap starts. Maturity = when it ends.
- "Hedge" / "hedging" / "swap" / "derivative" are related terms

CRITICAL RULES:
1. ONLY extract information EXPLICITLY stated in the document
2. Provide exact quotes as evidence
3. Use "NOT_FOUND" if clearly absent, "POSSIBLY_PRESENT" if likely in another section
4. Output valid JSON.""",

    "syndication_ing": """You are an expert financial analyst extracting syndication and lender data.

DOMAIN RULES:
- ING Bank N.V. / ING Group / ING = same entity, look for ANY ING reference
- ING share = ING commitment amount / Total facility amount × 100%
- If you can find ING's commitment AND total amount, CALCULATE the percentage
- "Syndicated" = multiple lenders. "Bilateral" = single lender. "Club deal" = small group.
- Look for lender schedules, commitment tables, participation lists

CRITICAL RULES:
1. ONLY extract information EXPLICITLY stated in the document
2. Provide exact quotes as evidence
3. Use "NOT_FOUND" if clearly absent, "POSSIBLY_PRESENT" if likely in another section
4. Output valid JSON.""",

    "cash_sweep": """You are an expert financial analyst extracting cash sweep mechanisms.

DOMAIN RULES:
- "Cash sweep" = mandatory prepayment from excess cash flow
- "Predetermined" = fixed % of excess cash (e.g., 50% of excess cash)
- "Targeted" = payments aimed at reaching a target debt balance
- Look for "excess cash flow", "mandatory prepayment", "cash waterfall", "priority of payments"
- The sweep percentage is how much of excess cash goes to debt repayment

CRITICAL RULES:
1. ONLY extract information EXPLICITLY stated in the document
2. Provide exact quotes as evidence
3. Use "NOT_FOUND" if clearly absent, "POSSIBLY_PRESENT" if likely in another section
4. Output valid JSON.""",

    "fees": """You are an expert financial analyst extracting fee structures.

DOMAIN RULES:
- Upfront fee = arrangement fee = structuring fee = front-end fee (paid at signing)
- Commitment fee = fee on UNDRAWN amounts (not on drawn amounts)
- Commitment fee is sometimes expressed as % of margin (e.g., "40% of the Applicable Margin")
- LC fee = letter of credit issuance fee
- Agency fee is NOT the same as upfront/commitment fee (exclude it)

CRITICAL RULES:
1. ONLY extract information EXPLICITLY stated in the document
2. Provide exact quotes as evidence
3. Use "NOT_FOUND" if clearly absent, "POSSIBLY_PRESENT" if likely in another section
4. Output valid JSON.""",
}

# =====================================================================
# PROMPT TEMPLATES
# =====================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert financial analyst specializing in project finance and credit facilities.
Your task is to extract specific information from financial documents with extreme precision.

CRITICAL RULES:
1. ONLY extract information that is EXPLICITLY stated in the document
2. For EVERY value you extract, you MUST provide the exact quote from the document as evidence
3. If information is clearly not present in the provided text, use "NOT_FOUND"
4. If the field is likely present in the full document but not in the provided excerpts, use "POSSIBLY_PRESENT" - this tells the system to search harder
5. Be precise with numbers, dates, percentages - copy them exactly as written
6. For Yes/No fields, only answer "Yes" or "No" based on explicit document content
7. Convert dates to MM/YYYY format when possible
8. Convert basis points to percentages (e.g., 250bps = 2.50%)
9. For ratios like 1.20:1, output as 1.20x

Output your response as valid JSON."""

EXTRACTION_USER_TEMPLATE = """Extract the following fields from this document section:

FIELDS TO EXTRACT:
{fields_description}

DOCUMENT TEXT:
---
{document_text}
---

Respond with a JSON object in this exact format:
{{
    "fields": {{
        "Field Name 1": {{
            "value": "extracted value or NOT_FOUND",
            "evidence": "exact quote from document that supports this value",
            "confidence": "HIGH/MEDIUM/LOW"
        }},
        "Field Name 2": {{
            "value": "...",
            "evidence": "...",
            "confidence": "..."
        }}
    }},
    "facilities_detected": ["list of distinct facility/tranche names if multiple found"],
    "notes": "any important observations about the extraction"
}}

IMPORTANT:
- Evidence must be a direct quote, not a paraphrase
- If a field has multiple values (e.g., multiple facilities), list them all
- Confidence is HIGH if explicitly stated, MEDIUM if inferred from context, LOW if uncertain
- Use "NOT_FOUND" ONLY when the information genuinely does not exist in this type of document
- Use "POSSIBLY_PRESENT" when the field likely exists in the full document but is not visible in the provided excerpts (this triggers a deeper search)"""

FACILITY_DETECTION_PROMPT = """Analyze this document and identify ALL distinct credit facilities or tranches.

For each facility found, provide:
1. Facility name
2. Type (Term Loan, RCF, LC, Bond, etc.)
3. Amount/Limit
4. Key identifying characteristics

DOCUMENT TEXT:
---
{document_text}
---

Respond with JSON:
{{
    "total_facilities": <number>,
    "facilities": [
        {{
            "name": "Facility A",
            "type": "Term Loan",
            "amount": "EUR 100,000,000",
            "characteristics": "Senior secured, 7-year tenor"
        }}
    ],
    "is_single_facility": true/false,
    "notes": "observations about document structure"
}}"""

VERIFICATION_PROMPT = """Verify the following extracted data against the source document.
Check for:
1. Accuracy - does the extracted value match the document?
2. Completeness - is there additional relevant information?
3. Consistency - do related fields make sense together?

EXTRACTED DATA:
{extracted_data}

SOURCE DOCUMENT:
{document_text}

Respond with JSON:
{{
    "verified_fields": {{
        "Field Name": {{
            "original_value": "...",
            "verified_value": "...",
            "is_correct": true/false,
            "correction_reason": "if incorrect, explain why"
        }}
    }},
    "missing_information": ["fields that could be filled but weren't"],
    "inconsistencies": ["any logical inconsistencies found"]
}}"""
