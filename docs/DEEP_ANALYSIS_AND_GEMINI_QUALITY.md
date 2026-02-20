# Deep Project Analysis & Path to Gemini 2.5 Pro–Level Quality

This document is a **standalone, very deep analysis** of Project Sunshine: architecture, data flow, quality definition, and how to get extraction results **close to Gemini 2.5 Pro**.

---

## 1. Project Purpose and “Close to Gemini” Definition

### 1.1 What the project does

Project Sunshine is a **document extraction pipeline** for **project finance** documents (facility agreements, term sheets, credit agreements, amendments). It:

- Takes an archive of documents (PDF, Word, Excel, etc.) per client/company.
- Preprocesses them (text extraction, optional OCR via VLM, structure detection, chunking).
- **Consolidates all files per company** into one logical document.
- Runs **multi-pass extraction** with a local LLM (Qwen3-14B, 4-bit): facility detection → per-facility field-group extraction → optional verification → deep extraction for missing fields → cross-reference checks.
- Outputs a **CSV** with 70+ structured fields (borrower, facility name, pricing, covenants, hedging, etc.) and **evidence-backed JSON** per company.

So the pipeline is a **retrieval-augmented, multi-pass, evidence-based extractor** that must match a **fixed schema** (same columns as used for Gemini comparison).

### 1.2 How “Gemini 2.5 Pro” is defined in this repo

- **Golden record**: Gemini 2.5 Pro extraction is the **reference**. It is ingested via `tools/convert_gemini.py`, which maps Gemini XLSX/CSV column names to the pipeline’s `ALL_FIELDS` and produces a **golden CSV** in the same schema.
- **Quality metric**: `src/quality_compare.py` compares pipeline CSV vs golden CSV:
  - **Overall accuracy** = (exact_match + fuzzy_match) / (exact + fuzzy + mismatch + golden_only).
  - **golden_only** = field where Gemini has a value but we don’t (missed extraction).
  - **mismatch** = both have a value but they differ.
  - Fuzzy matching: normalized text, numbers (tolerance), dates (MM/YYYY vs alternatives), ratios (1.20x vs 1.20:1), Yes/No.

So **“result close to Gemini 2.5 Pro”** here means:

1. **High overall accuracy** vs the converted golden CSV (exact + fuzzy).
2. **Low golden_only** (we don’t miss what Gemini found).
3. **Low mismatch** (when we extract, we match Gemini’s interpretation where the document supports it).
4. **Same schema and semantics** as the golden record (column mapping and normalization aligned with `convert_gemini` and `quality_compare`).

---

## 2. End-to-End Architecture (Deep Walkthrough)

### 2.1 Stage 1: Preprocessing (`preprocess_v2.py`)

- **Input**: Archive (e.g. ZIP) or extracted files; one “company” = one folder or manifest grouping.
- **Per file**:
  - PDF: PyMuPDF text + optional OCR (EasyOCR or **VLM Qwen2.5-VL-3B**) for pages that need it.
  - Word/Excel: python-docx, openpyxl, extract_msg, etc.
- **Structure**: Section headers, table detection (pipe/tab, numeric columns), so that chunking can respect boundaries.
- **Output**: Plain text files + **chunks** (token-based, with overlap). Chunk size/overlap from `config`: `CHUNK_SIZE=2000`, `CHUNK_OVERLAP=400`.
- **Manifest**: `manifest.json` lists company, original_file, text_file, chunks_file.

**Quality impact**: If OCR or text extraction drops or corrupts text (especially tables, numbers), the LLM never sees the right content. Multilingual docs use langdetect; VLM/EasyOCR affect non-English and scanned pages. **Tables** are critical for many fields (pricing, covenants, ING share); table detection is heuristic (pipe/tab, separators).

### 2.2 Stage 2: Consolidation (`consolidate.py`)

- **Per company**: All preprocessed text files are concatenated with `[SOURCE FILE: ...]` markers.
- **Chunking**: The **full combined text** is re-chunked with the same tokenizer (CHUNK_SIZE, CHUNK_OVERLAP) so that **every character is in at least one chunk** (full coverage).
- **Output**: `ConsolidatedDocument`: `full_text`, `chunks` (list of dicts with `text`, `token_count`, `source_files`), and metadata.

**Quality impact**: Company-level consolidation is correct for “one company = one deal”: information spread across multiple PDFs (e.g. facility in one file, pricing in another) is in one pool. Chunk boundaries can still split a sentence or table; overlap mitigates but does not eliminate this.

### 2.3 Retrieval (`retriever.py`)

- **Retriever types**: **BM25**, **embedding** (Qwen3-Embedding-0.6B), or **hybrid** (BM25 + embedding, configurable weight).
- **BM25**: Token-level + **financial synonym expansion** (`FINANCIAL_SYNONYMS`): e.g. “spread” → margin, applicable margin, etc. Improves recall on project-finance jargon.
- **Index**: Built from chunk **text** only (no separate title/metadata in the score).
- **Usage**:
  - **Facility detection**: Multiple targeted queries (English + multilingual) to gather facility-related sections; then a 5K-token budget is filled by scored candidates.
  - **Field-group extraction**: `retrieve_for_field_group(retriever, keywords, field_names, top_k=MAX_CHUNKS_PER_FIELD_GROUP)`. Combines `FIELD_GROUPS[group]["keywords"]` and words from field names into one query; returns top_k chunk texts.
  - **Deep extraction**: Per-field strategies in `deep_extract.py` (field-specific keywords, field name, description) with `find_best_context_for_field(..., top_k=6)`.

**Quality impact**: If the **best chunk for a field is never in top_k**, the LLM cannot extract it. So:
- **Recall of retrieval** directly caps extraction recall.
- BM25 is keyword/synonym-based; embedding is semantic. Hybrid can help when Gemini “sees” the whole doc (or long context) and we only send a few chunks.
- `MAX_CHUNKS_PER_FIELD_GROUP=3` and `top_k=3` in retrieval mean we send **at most 3 chunks per field group** per call (~6K tokens of doc). Gemini 2.5 Pro can use much longer context; we compensate with **section passes** (see below).

### 2.4 Extraction flow (`extract_v2.py`)

- **Model**: Qwen3-14B (or configured path), 4-bit quantized, optionally Flash Attention; loaded once per run.
- **Per company**:
  1. **Language**: `detect_document_language(full_text)` (langdetect) → used for multilingual facility queries and language hint in prompts.
  2. **Facility detection**: One LLM call with `FACILITY_DETECTION_PROMPT` and a **retrieval-built** context (5K tokens). Output: list of facilities (name, type, amount, characteristics). If parsing fails, fallback: one “Primary Facility”.
  3. **Per facility**: **extract_with_full_coverage(...)**:
     - For **each of 12 field groups**:
       - **BM25/semantic retrieval**: `retrieve_for_field_group` → up to `MAX_CHUNKS_PER_FIELD_GROUP` chunks.
       - **First pass**: `extract_field_group(group_name, group_config, relevant_chunks, facility_context, model, tokenizer)`. One LLM call per group with:
         - System: `FIELD_GROUP_SYSTEM_PROMPTS[group]` or generic `EXTRACTION_SYSTEM_PROMPT`.
         - User: `EXTRACTION_USER_TEMPLATE` (fields description + document text) + optional **few-shot** from `examples/few_shot_template.py` + language hint.
         - Output: JSON `{ "fields": { "Field Name": { "value", "evidence", "confidence" } }, ... }`.
       - **Section passes** (full coverage): If not “all fields HIGH confidence”, we consider section windows over **all** consolidated chunks. We now **cap** this at `MAX_SECTION_PASSES_PER_GROUP` (e.g. 2) to avoid 20+ calls per group. Section candidates that are already fully covered by the BM25 pass are skipped; then we run at most N section passes.
       - **Merge**: `merge_field_extractions(results, fields)` → for each field keep best confidence / evidence.
  4. **Verification** (optional, can skip with `--skip_verification`): One LLM call per facility with `VERIFICATION_PROMPT` (extracted data + document excerpt) to correct errors.
  5. **Deep extraction** (`deep_extract.py`): For fields that are NOT_FOUND, POSSIBLY_PRESENT, or low confidence, **per-field** LLM calls with targeted retrieval and optional pattern/table hints. Capped at `MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY` (e.g. 25).
  6. **Cross-reference** (`cross_reference_fields`): Rule-based consistency (e.g. if syndicated=Yes then ING share should be present; covenants=Yes vs ratio presence) and auto-fixes.

**Quality impact**:
- **Facility detection** errors (merge two facilities into one, or split one into two) propagate to per-facility rows and to row alignment in `quality_compare` (row-by-row by facility index).
- **Field-group prompts** and **few-shot** shape format and domain behavior (dates MM/YYYY, ratios 1.20x, bps→%, Yes/No). Missing or weak group prompts → more NOT_FOUND or wrong format.
- **Section pass cap**: Fewer section passes = lower recall (we may never send the chunk where the answer lives); more section passes = higher recall but slower and more cost.
- **Verification** can fix some mistakes but adds one call per facility.
- **Deep extraction** is the main safety net for missed fields; capping it at 25 per facility may leave some golden_only.
- **Cross-reference** fixes logical inconsistencies so that our output is more “consistent” and closer to how Gemini might answer (e.g. Covenants=Yes when ratios are present).

### 2.5 Output and normalization

- **extraction_to_rows**: Each facility → one CSV row; all `EXTRACTABLE_FIELDS` filled from `facility.fields`; `normalize_field_value(field_name, value)` applied.
- **normalize.py**: Per-field-type rules:
  - **Dates** → MM/YYYY (multiple input formats supported).
  - **Percentages** → X.XX% (including bps conversion).
  - **Ratios** → X.XXx (e.g. 1.20:1 → 1.20x).
  - **Yes/No** → strict Yes/No.
- **quality_compare** then compares these normalized values to the golden CSV (with its own normalization for comparison: lowercase, strip, date/ratio tolerance).

**Quality impact**: If Gemini outputs “2.5%” and we output “2.50%”, comparison can still be exact or fuzzy. If our **schema or semantics** differ from what `convert_gemini` expects (e.g. field meaning or units), we get mismatches or golden_only. The **column mapping** in `tools/convert_gemini.py` (GEMINI_TO_PIPELINE) is the contract: pipeline field names and intended semantics must match that mapping.

---

## 3. Where Quality Can Lag vs Gemini 2.5 Pro

### 3.1 Model and context

- **Gemini 2.5 Pro**: Very large context, strong instruction following, strong at tables and long documents.
- **Qwen3-14B 4-bit**: Smaller model, 8K input cap (we truncate at 8K), no native “full document” in one call. So we **depend heavily on retrieval** to bring the right passage to the model. Any retrieval miss is a potential golden_only or wrong value.

### 3.2 Retrieval recall

- We send **3 chunks per field group** by default (config: `MAX_CHUNKS_PER_FIELD_GROUP=3`). If the answer is in chunk 4 or 5, we never see it in the first pass.
- Section passes add coverage but are **capped** (e.g. 2) for speed; so some sections are never queried for that group.
- BM25/embedding may rank a chunk with the answer **below** top_k for that group’s query (e.g. different wording than keywords).

### 3.3 Prompt and instruction alignment

- **Field names and descriptions** in `FIELD_GROUPS` must match **Gemini’s intent** as encoded in the golden CSV and in `convert_gemini`. Any mismatch (e.g. “Credit limit ['000]” vs “Credit limit [ '000] (in accounting format with currency)”) is already handled by the converter; but **semantic** differences (what to put in the cell) can cause mismatch.
- **Domain rules** in `FIELD_GROUP_SYSTEM_PROMPTS` (e.g. “Applicable Margin = Spread”, “250bps = 2.50%”) align the model with project finance; if Gemini follows the same rules, we get closer. If our prompts are vaguer, we drift.
- **Few-shot** examples are short and single-example; Gemini may have seen many similar docs. More or better few-shots could narrow the gap.

### 3.4 Facility detection and row alignment

- **quality_compare** matches rows by **company** and then **facility index**. If we detect 2 facilities and Gemini has 3 (or the order differs), row alignment is wrong and accuracy drops.
- Facility detection uses 5K tokens of retrieval-built context; if the “facility list” is in a table we don’t retrieve well, we under-detect or mis-name.

### 3.5 Coverage vs speed trade-offs

- **MAX_SECTION_PASSES_PER_GROUP=0** → BM25 only → fastest, lowest recall.
- **MAX_SECTION_PASSES_PER_GROUP=2** → balanced.
- **MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY=25** → we may not deep-extract the 26th missing field that Gemini would fill.
- **--skip_verification** → faster but no verification pass to fix extraction errors.

### 3.6 Normalization and schema

- **normalize.py** must produce values that **quality_compare** treats as exact or fuzzy match to Gemini. E.g. dates MM/YYYY, ratios X.XXx, percentages X.XX%. If Gemini uses a different convention (e.g. “40% of margin” for commitment fee), we may need to either normalize the same way or extend `quality_compare` fuzzy logic.
- **convert_gemini** already maps Gemini columns to pipeline names; any new pipeline field or renamed field must be reflected there and in the golden CSV.

---

## 4. Recommendations to Get Close to Gemini 2.5 Pro

### 4.1 Measure first

- Run **quality_compare** with a golden CSV from Gemini (via `convert_gemini`) and your pipeline CSV.
- Inspect **per_field_accuracy** and **mismatches_detail**: which fields have low accuracy? Are they golden_only (we miss) or mismatch (we disagree)?
- Use this to decide whether to focus on **retrieval**, **prompts**, **facility detection**, or **normalization**.

### 4.2 Retrieval

- **Increase chunks per group** when GPU memory allows: e.g. `MAX_CHUNKS_PER_FIELD_GROUP=5` (or 6) so the model sees more context per field group.
- Prefer **hybrid** retriever (`--retriever hybrid`) so semantic similarity (embedding) complements BM25; especially helpful for long, varied wording.
- **Section passes**: For “quality first” runs, increase `MAX_SECTION_PASSES_PER_GROUP` (e.g. 4–5) so more of the document is seen per group; balance with runtime.
- Consider **field-specific retrieval** for weak groups: e.g. use `retrieve_specific_field` or deeper keyword lists for covenants, pricing, ING share.

### 4.3 Prompts and few-shot

- **Extend FIELD_GROUP_SYSTEM_PROMPTS** to all 12 groups (today only a subset have custom prompts). Use the same domain rules and output conventions (MM/YYYY, X.XXx, Yes/No, bps→%) so the model behaves like the golden schema.
- **Add or refine few-shot examples** in `examples/few_shot_template.py` for groups that underperform: e.g. commitment fee “40% of Applicable Margin”, ING share calculation, multiple facilities.
- **Align wording with Gemini**: If you have sample Gemini prompts or outputs, mirror phrases like “Credit limit in '000 in accounting format with currency” in field descriptions so our model targets the same thing.

### 4.4 Facility detection

- **Improve facility-detection context**: Add queries that target **tables** (e.g. “commitment table”, “lender schedule”, “facility schedule”) so the model sees facility lists.
- **Stricter JSON schema**: Ask for exact fields (name, type, amount) and optionally “order in document” so row order matches Gemini when possible.
- **Post-check**: If we detect 1 facility but the document is clearly multi-tranche (e.g. Term Loan A/B), consider a second pass or heuristics from chunk titles.

### 4.5 Deep extraction and verification

- For quality runs, **do not skip verification** (omit `--skip_verification`).
- **Increase** `MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY` (e.g. 35–40) if golden_only is high and many are “we didn’t try”.
- **Prioritize POSSIBLY_PRESENT** in deep extraction (already done); ensure fields that often appear in tables (e.g. ING share, commitment fee) have good **table extraction** and retrieval in `deep_extract.py`.

### 4.6 Normalization and comparison

- **Review** `normalize.py` for fields that often mismatch: e.g. commitment fee “40% of margin” vs “40%” vs “40% of Applicable Margin”. Add normalizations or **quality_compare** fuzzy rules so that equivalent answers count as match.
- **Extend** `quality_compare` if needed: e.g. synonym lists for categorical values (e.g. “Interest rate swap” vs “IRS”), or more date/ratio formats.
- Keep **convert_gemini** and **ALL_FIELDS** in sync: when you add or rename a pipeline field, update the converter and the golden CSV schema.

### 4.7 Preprocessing and tables

- **Tables**: Many golden values come from tables. Improve table detection in `preprocess_v2.py` (e.g. better heuristics or a small table-structure model) and consider **passing table blocks explicitly** to the retriever or to field-group prompts (e.g. “Below is a table from the document: ...”).
- **OCR**: For scanned PDFs, ensure VLM/EasyOCR is enabled and that the preprocessed text is clean enough for retrieval and LLM.

### 4.8 Model and inference

- **Larger or better model** (e.g. Qwen 32B, or same 14B with better quantization) can improve instruction following and table reasoning.
- **Temperature**: Already low (0.1) / greedy for extraction; keep it for stability.
- **Max new tokens**: Per-group limits in `FIELD_GROUP_MAX_TOKENS` are fine; ensure they are enough for the largest group’s JSON (e.g. covenants_ratios, hedging).

---

## 5. Summary: Quality Levers (Checklist)

| Lever | Where | Effect on “close to Gemini” |
|-------|--------|-----------------------------|
| Golden CSV + quality_compare | tools/convert_gemini, src/quality_compare.py | Defines “Gemini” and accuracy metric. |
| Retrieval type | config RETRIEVER_TYPE, --retriever | Hybrid/embedding can improve recall vs BM25. |
| Chunks per group | MAX_CHUNKS_PER_FIELD_GROUP | More chunks → more context → less golden_only. |
| Section passes | MAX_SECTION_PASSES_PER_GROUP | More passes → better coverage, slower. |
| Field-group system prompts | config FIELD_GROUP_SYSTEM_PROMPTS | All groups covered, same conventions as Gemini. |
| Few-shot | examples/few_shot_template.py | Better format and domain behavior. |
| Facility detection | extract_v2 detect_facilities, retrieval | Correct count and order → correct row alignment. |
| Verification | --skip_verification | On → fewer mismatches. |
| Deep extraction cap | MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY | Higher → fewer golden_only. |
| Normalization | src/normalize.py | Match Gemini format → more exact/fuzzy matches. |
| Table handling | preprocess_v2, deep_extract | Better tables → better numbers and lists. |
| Column mapping | tools/convert_gemini GEMINI_TO_PIPELINE | Schema and semantics aligned with golden. |

Using this checklist with **per-field and mismatch reports** from quality_compare will systematically close the gap to Gemini 2.5 Pro–level results.

---

## 6. One-Hour Budget: 100K Tokens per Company

**Constraint:** One company (average **~100,000 tokens** consolidated) must be processed in **≤ 1 hour**, while still targeting quality close to Gemini 2.5 Pro.

### 6.1 Time and call budget

- **Assumptions:** A100 20 GB; Qwen3-14B 4-bit; ~100K tokens → ~**62 chunks** (CHUNK_SIZE=2000, overlap 400). Preprocessing (and VLM) assumed done or negligible; the hour is for **extraction phase** (consolidation + retrieval + LLM calls).
- **Per-call time:** ~20–35 s per LLM call (depends on input/output length and Flash Attention). Use **~25 s** as planning average.
- **Budget:** 60 min − 5 min (consolidation, retrieval, IO) = **55 min** for LLM → **~132 calls** max at 25 s/call. To stay safely under 1 h, target **≤ 120 calls** per company.

**Call formula:**

- Facility detection: **1**
- Per facility: **12 × (1 + S)** extraction calls (S = section passes per group) + **V** verification (0 or 1) + **D** deep-extract calls (capped).
- **Total:** `1 + N_fac × (12×(1+S) + V + D)`.

Examples (N_fac = number of facilities):

| N_fac | S | V | D  | Total calls | ~Time @25s |
|-------|---|---|-----|-------------|------------|
| 2     | 1 | 0 | 20 | 89          | ~37 min    |
| 3     | 1 | 0 | 20 | 133         | ~55 min    |
| 4     | 1 | 0 | 15 | 1 + 4×39 = 157 → **too high** | — |
| 4     | 0 | 0 | 15 | 1 + 4×27 = **109** | ~45 min |
| 2     | 2 | 1 | 25 | 1 + 2×62 = 125 | ~52 min |

So for **≤ 1 h** with 2–3 facilities we can afford **S=1**, **V=0** (skip verification), **D=20**. For **4+ facilities** we need **S=0** and/or lower **D** (e.g. 15).

### 6.2 Recommended config for “1 h + Gemini-like quality”

Use these settings so that **one company (~100K tokens) stays under 1 hour** while keeping quality as high as possible:

| Setting | Value | Where | Reason |
|---------|--------|--------|--------|
| **MAX_SECTION_PASSES_PER_GROUP** | **1** | `config.py` | One extra section pass per group (BM25 + 1 section) keeps recall reasonable but caps calls at 12×2 = 24 per facility. |
| **MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY** | **20** | `config.py` | Prioritise POSSIBLY_PRESENT; 20 is enough for most missing-field recovery without blowing the budget. |
| **Verification** | **Skip** | CLI `--skip_verification` | Saves 1 call per facility; small quality trade-off. |
| **MAX_CHUNKS_PER_FIELD_GROUP** | **3** or **4** | `config.py` | 3 = smaller prompts (faster). 4 = more context per call (fewer section passes needed in theory but longer calls). Keep **3** for strict 1 h. |
| **Retriever** | **BM25** | `config.py` or `--retriever bm25` | No embedding load/index; faster startup and no GPU contention with LLM. Use **hybrid** only if you have proven quality gain and can afford indexing time. |
| **Flash Attention** | **On** | `--flash_attention` | Faster inference; use if available on A100. |

**Suggested command (extraction only, one company):**

```bash
python -m src.main --stage extract --skip_preprocess --flash_attention --skip_verification --company "CompanyName" --retriever bm25
```

**Config snippet for 1-hour budget** (in `src/config.py`):

```python
# One-hour budget (~100K tokens per company): keep extraction ≤ ~120 LLM calls
MAX_SECTION_PASSES_PER_GROUP = 1   # 1 section pass per group (BM25 + 1)
MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY = 20
# Use --skip_verification on CLI
```

### 6.3 If you have more than ~3 facilities per company

- Set **MAX_SECTION_PASSES_PER_GROUP = 0** (BM25 only) and **MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY = 15** so that `1 + N_fac × (12 + 15)` stays ≤ 120 for N_fac ≤ 4 (e.g. 1 + 4×27 = 109).
- Or accept slightly over 1 h for companies with many facilities and keep S=1, D=20 for better quality on typical 2–3 facility cases.

### 6.4 Quality vs speed (same 1 h budget)

- **Quality-first within 1 h:** S=1, D=20, verification off, BM25, good prompts + few-shot (as in §4). This is the best compromise.
- **Faster (e.g. 45 min):** S=0, D=15, `--skip_verification`, BM25 → fewer calls, lower recall.
- **Do not** increase S to 2 or turn verification on without reducing D or accepting >1 h for large N_fac.
