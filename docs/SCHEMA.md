# Extraction Schema Documentation

This document describes all fields extracted by Project Sunshine.

## Field Groups Overview

| # | Group | Fields | Purpose |
|---|-------|--------|---------|
| 1 | basic_info | 6 | Core facility identification |
| 2 | sponsor_info | 5 | Sponsor and ownership structure |
| 3 | project_details | 6 | Project characteristics |
| 4 | construction_guarantees | 5 | Construction phase protections |
| 5 | revenue_mitigants | 6 | Revenue risk mitigation |
| 6 | covenants | 12 | Financial covenants |
| 7 | syndication_ing | 3 | Syndication structure |
| 8 | dates_schedules | 7 | Timing and schedules |
| 9 | pricing | 6 | Interest rate structure |
| 10 | hedging | 8 | Hedging arrangements |
| 11 | cash_sweep | 5 | Prepayment mechanisms |
| 12 | fees | 3 | Fee structure |

---

## Detailed Field Definitions

### 1. Basic Information

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Borrower name | Text | Legal name of borrowing entity | "SolarCo Holdings B.V." |
| Senior facility name | Text | Name of senior credit facility | "Senior Secured Facilities" |
| Facility name | Text | Specific tranche name | "Term Loan A" |
| Type of tranche | Enum | Term Loan / RCF / LC / Bond | "Term Loan" |
| Currency of the facility | Code | ISO currency code | "EUR" |
| Credit limit ['000] | Amount | Total commitment with currency | "EUR 250,000,000" |

### 2. Sponsor Information

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Is a Sponsor linked to the project? | Yes/No | Presence of sponsor | "Yes" |
| Sponsor Name | Text | Name of sponsor entity | "Global Infrastructure Partners" |
| Sponsor ownership [%] | Percentage | Equity stake | "75%" |
| Is a HoldCo linked to the project? | Yes/No | Holding company presence | "Yes" |
| Name of the HoldCo | Text | Holding company name | "SolarCo Group Ltd" |

### 3. Project Details

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Project Sector (NACE Rev. 2) | Code | NACE classification | "D 35.11 - Production of electricity" |
| Project Sector (Aggregated) | Enum | Must be one of: Energy / Infrastructure / LNG / Manufacturing / Mining / TM / Wind and Solar | "Energy" |
| Project location [Country] | Text | Country name | "Spain" |
| Readiness | Enum | In operation / Construction needed | "Construction needed" |
| If construction needed: Readiness year | Year | Expected completion year | "2027" |
| If construction needed: Readiness month | Month | Expected completion month | "06" |

### 4. Construction Guarantees

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| If completion guarantees specified? | Yes/No | Presence of guarantees | "Yes" |
| Guarantee Development sponsors name | Text | Guarantor name | "Global Infrastructure Partners" |
| If D&C Contractor specified? | Yes/No | Contractor presence | "Yes" |
| Is there a fixed price, date certain, turnkey contract? | Yes/No | EPC structure | "Yes" |
| Name of the D&C Contractor | Text | Contractor or guarantor name | "EPC Solutions Inc." |

### 5. Revenue Mitigating Factors

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| If revenue mitigating factors specified? | Yes/No | Presence of revenue protection | "Yes" |
| Type of mitigating factor | Enum | Guaranteed price / Guaranteed add-on / Fix revenue component / Regulated rate-of-return / Variable OPEX passed through | "Guaranteed price (not in '000)" |
| Contractual or regulatory factor guarantor | Text | Entity providing guarantee | "Spanish Government" |
| Revenue mitigating factors applied to | Enum | all revenues / revenues of the borrower | "all revenues" |
| Volume constraint type | Enum | Absolute volume / % of volume covered | "% of volume covered" |
| Volume covered [%] | Percentage | Coverage percentage | "90%" |

### 6. Financial Covenants

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| If DSRA specified? | Yes/No | Debt Service Reserve Account | "Yes" |
| If covenants specified? | Yes/No | Presence of financial covenants | "Yes" |
| Minimum cash, '000 | Amount | Cash covenant threshold | "EUR 5,000,000" |
| Minimum Equity Reserve Account, '000 | Amount | Equity reserve threshold | "N/A" |
| Backward-looking DSCR (B-DSCR) | Ratio | Historical coverage ratio | "1.20x" |
| Forward-looking DSCR (F-DSCR) | Ratio | Projected coverage ratio | "1.25x" |
| Loan Life Coverage Ratio (LLCR) | Ratio | LLCR covenant | "1.30x" |
| Interest Coverage Ratio (ICR) | Ratio | ICR covenant | "N/A" |
| Net Debt / EBITDA | Ratio | Leverage covenant | "N/A" |
| Dividend Cover Ratio | Ratio | Dividend coverage | "N/A" |
| Net Debt / RAB | Ratio | Regulatory asset base ratio | "N/A" |
| Debt / Equity | Ratio | D/E ratio | "N/A" |

### 7. Syndication & ING Participation

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Is facility syndicated? | Yes/No | Multiple lenders | "Yes" |
| If ING commitment specified? | Yes/No | ING participation | "Yes" |
| ING share [%] | Calculated | ING percentage (amount / total) | "20% (EUR 50M / EUR 250M)" |

### 8. Dates & Schedules

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Inception date [MM/YYYY] | Date | Facility start date | "03/2024" |
| Maturity date [MM/YYYY] | Date | Facility end date | "03/2031" |
| Is tranche expected to be refinanced? | Yes/No | Refinancing expectation | "No" |
| Principal drawdown schedule | Text | Drawdown description | "Per project milestones" |
| Principal drawdown schedule [Date: %] | Schedule | Specific drawdowns | "06/2024: 30%; 12/2024: 40%; 06/2025: 30%" |
| Principal repayment schedule | Text | Repayment description | "Semi-annual amortization" |
| Principal repayment schedule [Date: %] | Schedule | Specific repayments | "03/2027: 5%; 09/2027: 5%..." |

### 9. Pricing

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Base Rate of the facility | Text | Reference rate | "EURIBOR" |
| Tenor of the base rate | Period | Interest period | "6M" |
| Fix interest rate [%] | Percentage | Fixed rate (if applicable) | "N/A" |
| Spread [%] | Percentage | Margin over base rate | "2.50%" |
| Interest rate floor % | Percentage | Floor rate | "0.00%" |
| Interest rate cap % | Percentage | Cap rate | "N/A" |

### 10. Hedging

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Is the facility hedged? | Yes/No | Hedging presence | "Yes" |
| How is the facility hedged | Enum | interest rate swaps / FX swaps / inflation swaps | "interest rate swaps" |
| Hedging: effective date | Date | Hedge start | "03/2024" |
| Hedging: fixed rate | Percentage | Swap fixed rate | "3.25%" |
| Hedging: maturity date | Date | Hedge end | "03/2031" |
| Hedging: notional | Amount | Hedge amount | "EUR 200,000,000" |
| Hedging: spread | Percentage | Hedge spread | "N/A" |
| Hedging: % of exposure hedged | Percentage | Coverage ratio | "80%" |

### 11. Cash Sweep

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Is there a cash sweep mechanism? | Yes/No | Sweep presence | "Yes" |
| Cash sweep structure | Enum | Predetermined / Targeted debt balance / Transmission asset linked | "Predetermined (as % of cash available)" |
| For predetermined - additional cash [%] | Percentage | Sweep percentage | "50%" |
| For targeted - target repayment profile [%] | Percentage | Target percentage | "N/A" |
| For transmission - sweep percentage [%] | Percentage | Asset-linked sweep | "N/A" |

### 12. Fees

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| Upfront Fee [%] | Percentage | Initial fee | "1.00%" |
| Commitment fee [%] | Percentage | Undrawn fee | "0.40%" |
| Letter of credit issuance fee [%] | Percentage | LC fee | "N/A" |

---

## Confidence Levels

| Level | Definition |
|-------|------------|
| HIGH | Value explicitly stated in document |
| MEDIUM | Value inferred from context or calculated |
| LOW | Value uncertain or partially matched |

---

## Special Values

| Value | Meaning |
|-------|---------|
| NOT_FOUND | Field not present in document |
| N/A | Field not applicable to this facility |
| EXTRACTION_ERROR | Technical error during extraction |

---

## Customization

To add new fields:

1. Add to appropriate group in `src/config.py` → `FIELD_GROUPS`
2. Add to `ALL_FIELDS` list
3. Add keywords for BM25 retrieval
4. Optionally add pattern matching in `src/deep_extract.py`
