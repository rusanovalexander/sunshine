# A100 20 GB Speed Optimization Guide

**Problem:** One company currently runs ~5 hours on A100 20 GB, which is not acceptable.

**Root cause:** The pipeline is dominated by **hundreds of sequential LLM calls per company** with no batching and aggressive “full coverage” section passes. Each call runs one at a time on the GPU.

---

## 1. Where Time Goes (Per Company)

| Stage | LLM calls (approx.) | Notes |
|-------|---------------------|--------|
| Facility detection | 1 | One call per company |
| **Field group extraction** | **12 groups × (1 BM25 + N section passes)** | **N was unbounded** (e.g. 15–30 section passes per group for large docs) |
| Verification | 1 per facility | Optional |
| **Deep extraction** | **1 per missing/POSSIBLY_PRESENT field per facility** | Often 20–50+ per facility |

**Example:** 2 facilities, 60 chunks, 40 missing fields per facility:
- Before: 1 + 2×(12×1 + 12×20 section + 1 verify) + 2×40 ≈ **550+ LLM calls** → ~5 h at ~30 s/call.
- After caps: 1 + 2×(12×1 + 12×2 section + 1 verify) + 2×25 ≈ **115 LLM calls** → target **<1.5 h**.

---

## 2. Implemented Optimizations

### 2.1 Cap section passes (largest impact)

**Config:** `src/config.py`

```python
MAX_SECTION_PASSES_PER_GROUP = 2   # Max extra section passes per field group (0 = BM25 only)
```

- **Before:** For each of 12 field groups, the pipeline did one BM25 pass plus a section loop over **all** chunk windows (e.g. 20+ passes for 60 chunks with `MAX_CHUNKS_PER_FIELD_GROUP=3`).
- **After:** At most **2** additional section passes per field group. Section candidates are still chosen so that sections already fully covered by BM25 are skipped.
- **Effect:** Cuts the bulk of “full coverage” LLM calls (often 200+ per company down to ~24), giving a **~3–4×** speedup for the extraction stage.

### 2.2 Cap deep extraction per facility

**Config:** `src/config.py`

```python
MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY = 25
```

- Deep extraction runs one LLM call per missing/POSSIBLY_PRESENT field. With 70+ fields, that could be 50+ calls per facility.
- **Effect:** Limits to 25 fields per facility (POSSIBLY_PRESENT first), reducing deep-extract time significantly.

### 2.3 Optional verification pass

**CLI:** `--skip_verification`

- Skips the verification LLM call per facility.
- **Use when:** Throughput is more important than the small quality gain from verification.

### 2.4 Flash Attention for BnB 4-bit model

**Code:** `src/gpu_optimizer.py` – when loading saved BnB 4-bit checkpoints, `use_flash_attention=True` now uses `attn_implementation="flash_attention_2"` on A100, with fallback to `sdpa` if Flash Attention is not available.

- Run with: `python -m src.main --stage extract --flash_attention ...`
- **Effect:** Faster attention and better GPU utilization per call.

---

## 3. Recommended Run for Speed (A100 20 GB)

```bash
# Pre-quantize once (then set MODEL_PATH to the _bnb4 path in config)
python -m src.main --stage quantize

# In config.py set MODEL_PATH to the pre-quantized path, then:
python -m src.main --stage all --flash_attention --skip_verification --company "CompanyName"
```

Tuning in `config.py` for maximum speed (at the cost of some coverage):

- `MAX_SECTION_PASSES_PER_GROUP = 0` — BM25 only, no section passes (~2× faster than 2, lower coverage).
- `MAX_SECTION_PASSES_PER_GROUP = 1` — One section pass per group (good balance).
- `MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY = 15` — Fewer deep extraction calls.

---

## 4. Further Optimizations (Not Yet Implemented)

### 4.1 Batched inference

- **Current:** Every `llm_generate()` is a single sequence; `BatchInferenceEngine` in `gpu_optimizer.py` is not used in the extraction path.
- **Idea:** Batch 2–4 independent prompts (e.g. 2–4 field groups or 2–4 deep-extract fields) with padding and run one `model.generate()` with batch size > 1. This can give ~1.5–2× GPU throughput if memory allows.
- **Constraint:** A100 20 GB with 4-bit 14B may allow batch size 2 for moderate context; needs testing.

### 4.2 Fewer field groups per run

- Merge small field groups so that fewer “group passes” are needed (e.g. 12 → 8). Fewer groups ⇒ fewer BM25 + section passes.

### 4.3 Disable or throttle debug traces

- Each LLM call writes a JSON trace under `EXTRACTION_DIR/debug_traces/`. For production, disable or write asynchronously to reduce I/O and minor CPU overhead.

### 4.4 Smaller/faster model

- Use a 7B model (e.g. Qwen2.5-7B) with same schema for lower latency per call and possibility of batch size 2–4 on 20 GB.

### 4.5 vLLM or TGI

- Replace `model.generate()` with vLLM or Text Generation Inference for continuous batching and higher GPU utilization. Requires refactoring the extraction loop to use an HTTP or queue-based API.

---

## 5. Summary

| Change | Config / CLI | Expected impact |
|--------|----------------|------------------|
| Cap section passes | `MAX_SECTION_PASSES_PER_GROUP = 2` | **~3–4×** fewer extraction calls |
| Cap deep extraction | `MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY = 25` | Large reduction in deep-extract time |
| Skip verification | `--skip_verification` | 1 call less per facility |
| Flash Attention | `--flash_attention` | Faster per-call inference on A100 |
| Pre-quantized model | `--stage quantize` then use _bnb4 path | Faster load and less peak memory |

**Target:** With these changes, one company should drop from **~5 h** to roughly **1–1.5 h** on A100 20 GB. For even faster runs, set `MAX_SECTION_PASSES_PER_GROUP = 0` and optionally reduce `MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY` and use `--skip_verification`.

**Strict 1-hour budget (e.g. ~100K tokens per company):** For a hard **≤ 1 hour per company** with quality close to Gemini 2.5 Pro, use the preset in `config.py` (MAX_SECTION_PASSES_PER_GROUP=1, MAX_DEEP_EXTRACT_FIELDS_PER_FACILITY=20), run with `--skip_verification --retriever bm25 --flash_attention`, and see **docs/DEEP_ANALYSIS_AND_GEMINI_QUALITY.md** §6 for the full time model and recommendations.
