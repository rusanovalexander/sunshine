# Implementation Checklist: Pre-Embed at Preprocessing

Exact list of what will be changed. Implement in this order.

---

## 0. Execution time estimate (before implementation)

Assumptions: one company ≈ 100,000 tokens (~62 chunks with CHUNK_SIZE=2000, overlap 400). Qwen3-Embedding-0.6B; A100 20 GB or CPU.

### Preembed stage (one-time per company)

| Step | GPU (A100) | CPU only |
|------|------------|----------|
| Load tokenizer | ~2–5 s (once) | same |
| Load embedding model | ~15–30 s (once per run) | same |
| Per company: consolidate (merge + chunk) | ~2–5 s | same |
| Per company: save 3 files | ~1 s | same |
| Per company: encode ~62 chunks (batches of 32) | ~10–25 s | ~2–5 min |
| Per company: save _EMBEDDINGS.pt | <1 s | same |
| **First company total** | **~45–70 s** | **~3–6 min** |
| **Each extra company** | **~15–35 s** | **~2–5 min** |

So for **one company (100K tokens)** on GPU: preembed ≈ **0.5–1.5 min**. For 10 companies: ~5–15 min (one model load + 10 × per-company). Preembed is typically run once after preprocessing; it does not need to run in the 1-hour extraction window.

### Extraction: with vs without prebuilt

| | Without prebuilt (current) | With prebuilt |
|--|----------------------------|----------------|
| Per company: build retriever | Consolidate ~5 s + encode 62 chunks on GPU ~15–25 s → **~20–30 s** | Load consolidated from disk ~1 s + load _EMBEDDINGS.pt ~0.5 s → **~1–2 s** |
| **Saved per company at extraction** | — | **~18–28 s** |

So with prebuilt, extraction avoids document encoding; you save **~20–30 s per company** when using embedding/hybrid retriever. That keeps the 1-hour budget easier to meet (e.g. 2–3 min saved for one company, more for many companies). Query encoding (one short vector per retrieval call) stays and is cheap (~0.05–0.1 s per query; tens of queries per company → a few seconds total).

### Summary

- **Preembed:** ~0.5–1.5 min per company on GPU (one-time, can run offline).
- **Extraction with prebuilt:** ~20–30 s faster per company (no document encoding).
- **Net:** Preembed cost is paid once; every later extraction run for that company is faster.

---

## 1. `src/config.py`

| What | Detail |
|------|--------|
| Add constant | `CONSOLIDATED_PREEMBED_SUBDIR = "consolidated"` — subdir under `PREPROCESSED_DATA_DIR` where prebuilt per-company dirs live. |
| Add constant | `USE_PREBUILT_EMBEDDINGS_IF_AVAILABLE = True` — extraction will prefer prebuilt when present. |
| Helper | Prebuilt dir for a company: `os.path.join(PREPROCESSED_DATA_DIR, CONSOLIDATED_PREEMBED_SUBDIR, safe_company_name)` where `safe_company_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company)`. |

No new CLI here; CLI in main.py.

---

## 2. `src/consolidate.py`

| What | Detail |
|------|--------|
| Import config | At top: `from .config import CHUNK_SIZE, CHUNK_OVERLAP` (or pass as args). |
| Refactor | In `consolidate_company_documents()`, call `create_comprehensive_chunks(..., chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)` instead of hardcoded 2000, 400. |
| New function | `load_consolidated_document(company_dir: str) -> Optional[ConsolidatedDocument]`. Reads `company_dir/_COMBINED_DOCUMENT.txt`, `_COMBINED_CHUNKS.json`, `_COMBINED_META.json`. Rebuilds `ConsolidatedDocument(company=..., full_text=..., source_files=..., source_references=..., total_chars=..., total_tokens=..., chunks=...)`. Return None if any file missing. |
| Optional | Add `save_consolidated_document_to_dir(doc, target_dir)` that saves to an arbitrary `target_dir` (for preembed saving to `PREPROCESSED_DATA_DIR/consolidated/safe_company/`), or have preembed call existing `save_consolidated_document(doc, base_dir)` with `base_dir = PREPROCESSED_DATA_DIR` and then move/rename, or pass a full path. Easiest: add a helper that takes `target_company_dir` and writes the three files there (same content as save_consolidated_document but dir is `target_company_dir` not `output_dir/doc.company`). So we need to save to `os.path.join(PREPROCESSED_DATA_DIR, "consolidated", safe_company)` — that's a different structure (no company subdir inside EXTRACTION_DIR). So new helper: `save_consolidated_document_to_path(doc, target_company_dir: str)` that writes `_COMBINED_DOCUMENT.txt`, `_COMBINED_CHUNKS.json`, `_COMBINED_META.json` into `target_company_dir`. |

---

## 3. `src/retriever.py`

| What | Detail |
|------|--------|
| New class method | `EmbeddingRetriever.from_pretrained_index(documents: List[str], embeddings_path: str, model, tokenizer) -> EmbeddingRetriever`. Load tensor from `embeddings_path` (`.pt` or `.npy`). If `.npy`, convert to torch tensor. Ensure L2-normalized (same as after index()). Set `self.documents = documents`, `self.doc_embeddings = tensor` (on CPU). Set `self.model`, `self.tokenizer` so `search()` can encode queries. Do not call `index()` or `_encode()` for documents. Return the instance. |
| Extend factory | `create_retriever_from_chunks(chunks, retriever_type, embedding_model, embedding_tokenizer, bm25_weight=0.5, prebuilt_embeddings_path: Optional[str] = None)`. |
| Embedding path | If `retriever_type == "embedding"` and `prebuilt_embeddings_path` is not None and `os.path.exists(prebuilt_embeddings_path)`: build `texts = [c.get('text','') for c in chunks]`, then `return EmbeddingRetriever.from_pretrained_index(texts, prebuilt_embeddings_path, embedding_model, embedding_tokenizer)`. Else current behaviour (retriever.index(texts)). |
| Hybrid path | If `retriever_type == "hybrid"` and `prebuilt_embeddings_path` is not None and exists: build BM25, build embedding retriever via `from_pretrained_index(texts, prebuilt_embeddings_path, ...)`, return `HybridRetriever(bm25, emb, bm25_weight)`. Do not call `emb.index(texts)` or `retriever.index(texts)`. Else current behaviour (both index). |

---

## 4. New file: `src/preembed.py`

| What | Detail |
|------|--------|
| Entrypoint | `def run_preembed(company_filter: Optional[str] = None) -> None`. |
| Steps | 1) Load manifest from `PREPROCESSED_DATA_DIR/manifest.json`. 2) Load tokenizer: `AutoTokenizer.from_pretrained(MODEL_PATH)` (from config). 3) Companies = `get_all_companies(manifest)`; if company_filter, filter to that company. 4) Create base dir `os.path.join(PREPROCESSED_DATA_DIR, CONSOLIDATED_PREEMBED_SUBDIR)`; `os.makedirs(..., exist_ok=True)`. 5) Load embedding model once: `load_embedding_model(EMBEDDING_MODEL_PATH)`. 6) For each company: a) `consolidated = consolidate_company_documents(company, manifest, PREPROCESSED_DATA_DIR, tokenizer)`. b) If None or no chunks, skip. c) `safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', company)`. d) `company_dir = os.path.join(PREPROCESSED_DATA_DIR, CONSOLIDATED_PREEMBED_SUBDIR, safe_name)`; `os.makedirs(company_dir, exist_ok=True)`. e) Save consolidated: write `_COMBINED_DOCUMENT.txt`, `_COMBINED_CHUNKS.json`, `_COMBINED_META.json` into company_dir (reuse logic from consolidate or call a new save helper that takes target_company_dir). f) Encode chunks: `texts = [c['text'] for c in consolidated.chunks]`. Encode in batches using same logic as EmbeddingRetriever (tokenizer max_length from config EMBEDDING_MAX_LENGTH, normalize). g) Stack and L2-normalize; save tensor as `os.path.join(company_dir, "_EMBEDDINGS.pt")` with `torch.save(embeddings.cpu(), path)`. 7) Optional: unload embedding model and log GPU memory. |
| Imports | consolidate (consolidate_company_documents, get_all_companies), retriever (load_embedding_model), config (PREPROCESSED_DATA_DIR, CONSOLIDATED_PREEMBED_SUBDIR, MODEL_PATH, EMBEDDING_MODEL_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MAX_LENGTH, EMBEDDING_BATCH_SIZE). Use EmbeddingRetriever's _encode logic or a shared helper for encoding + normalization. |

---

## 5. `src/main.py`

| What | Detail |
|------|--------|
| Stage choice | In parser, add `'preembed'` to `--stage` choices (e.g. `choices=['all', 'preprocess', 'preembed', 'extract', 'report', 'compare', 'quantize']`). |
| Run preembed | In main try block: if `args.stage == 'preembed'`: call `run_preembed(args.company)` (new function that imports and calls preembed.run_preembed). Then return or continue as appropriate (preembed is standalone, so after run_preembed return 0). |
| New function | `def run_preembed(args)`: from .preembed import run_preembed; run_preembed(company_filter=getattr(args, 'company', None)). |
| Pass prebuilt path to extraction | Extraction is in run_extraction; it calls process_company_consolidated from extract_v2. So the "check for prebuilt and load consolidated" lives in extract_v2.process_company_consolidated. main.py only needs to pass a flag or nothing (extract_v2 will check USE_PREBUILT_EMBEDDINGS_IF_AVAILABLE and look under PREPROCESSED_DATA_DIR/consolidated/safe_company). So no change in main for passing path — extract_v2 does the check. Optional: add CLI `--no-prebuilt-embeddings` to disable use of prebuilt so extraction always consolidates in memory; then pass that into run_extraction and down to process_company_consolidated. |

---

## 6. `src/extract_v2.py`

| What | Detail |
|------|--------|
| Imports | Add `load_consolidated_document` from .consolidate. Add from config: `PREPROCESSED_DATA_DIR`, `CONSOLIDATED_PREEMBED_SUBDIR`, `USE_PREBUILT_EMBEDDINGS_IF_AVAILABLE`. |
| Safe company name | Define or import a helper: `safe_company_name(company: str) = re.sub(r'[^a-zA-Z0-9_-]', '_', company)`. |
| In `process_company_consolidated` | At the start, after set_debug_company: if USE_PREBUILT_EMBEDDINGS_IF_AVAILABLE and retriever_type in ('embedding', 'hybrid'): compute `prebuilt_dir = os.path.join(PREPROCESSED_DATA_DIR, CONSOLIDATED_PREEMBED_SUBDIR, safe_company_name(company))`. Check `os.path.exists(os.path.join(prebuilt_dir, "_COMBINED_CHUNKS.json"))` and `os.path.exists(os.path.join(prebuilt_dir, "_EMBEDDINGS.pt"))`. If both exist: `consolidated = load_consolidated_document(prebuilt_dir)`. If consolidated is not None, skip calling `consolidate_company_documents(...)` and skip the "Save consolidated document for debugging" to EXTRACTION_DIR (or still save a copy to EXTRACTION_DIR for consistency). Set `prebuilt_embeddings_path = os.path.join(prebuilt_dir, "_EMBEDDINGS.pt")`. When calling `create_retriever_from_chunks(consolidated.chunks, ..., prebuilt_embeddings_path=prebuilt_embeddings_path)`. Else (no prebuilt or not embedding/hybrid): current behaviour — consolidate in memory, no prebuilt_embeddings_path. |
| create_retriever_from_chunks call | Add argument `prebuilt_embeddings_path=None`. When prebuilt was loaded, pass that path. |

---

## 7. Files not changed

- `deep_extract.py` — no change.
- `quality_compare.py`, `normalize.py`, `preprocess_v2.py` (except if we add a call to preembed from there; we're not), `gpu_optimizer.py` — no change.
- `examples/`, `tools/` — no change.

---

## 8. Conventions (reminder)

- Prebuilt dir: `PREPROCESSED_DATA_DIR / "consolidated" / safe_company_name /`
- Files: `_COMBINED_DOCUMENT.txt`, `_COMBINED_CHUNKS.json`, `_COMBINED_META.json`, `_EMBEDDINGS.pt`
- `_EMBEDDINGS.pt`: PyTorch tensor, shape `(num_chunks, embed_dim)`, float32, L2-normalized (same as EmbeddingRetriever after index()).
- Chunk order in `_COMBINED_CHUNKS.json` must match the order of rows in `_EMBEDDINGS.pt` (row i = embedding of chunk i).

---

## 9. Order of implementation

1. **config.py** — add CONSOLIDATED_PREEMBED_SUBDIR, USE_PREBUILT_EMBEDDINGS_IF_AVAILABLE.
2. **consolidate.py** — CHUNK_SIZE/CHUNK_OVERLAP from config; `load_consolidated_document(company_dir)`; optional `save_consolidated_document_to_path(doc, target_company_dir)` for preembed to use.
3. **retriever.py** — `EmbeddingRetriever.from_pretrained_index(...)`; `create_retriever_from_chunks(..., prebuilt_embeddings_path=...)` for embedding and hybrid.
4. **preembed.py** — new module with run_preembed() implementing the loop.
5. **main.py** — add --stage preembed and run_preembed; optional --no-prebuilt-embeddings.
6. **extract_v2.py** — check prebuilt dir, load_consolidated_document, pass prebuilt_embeddings_path into create_retriever_from_chunks.
