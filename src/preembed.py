"""
Pre-embed at preprocessing: consolidate per company and save chunk embeddings
so extraction can skip document encoding when using embedding/hybrid retriever.
"""

import os
import json
import logging
from typing import Optional

import torch
from transformers import AutoTokenizer

from .config import (
    PREPROCESSED_DATA_DIR,
    CONSOLIDATED_PREEMBED_SUBDIR,
    MODEL_PATH,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_BATCH_SIZE,
    get_prebuilt_company_dir,
)
from .consolidate import (
    consolidate_company_documents,
    get_all_companies,
    save_consolidated_document_to_path,
)
from .retriever import load_embedding_model, EmbeddingRetriever

logger = logging.getLogger(__name__)


def _encode_chunks_batched(
    texts: list,
    model,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 2048,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode texts in batches and return L2-normalized tensor (num_texts, embed_dim)."""
    retriever = EmbeddingRetriever(
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = retriever._encode(batch, max_length=max_length)
        all_embeddings.append(embs.cpu())
    out = torch.cat(all_embeddings, dim=0)
    out = torch.nn.functional.normalize(out, p=2, dim=1)
    return out


def run_preembed(company_filter: Optional[str] = None) -> None:
    """
    For each company: consolidate documents, save to consolidated/safe_company/,
    encode chunks with the embedding model, save _EMBEDDINGS.pt.
    """
    manifest_path = os.path.join(PREPROCESSED_DATA_DIR, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    logger.info("Loading tokenizer for consolidation chunking...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    companies = get_all_companies(manifest)
    if company_filter:
        companies = [c for c in companies if c == company_filter]
        if not companies:
            logger.warning(f"No company matching filter: {company_filter}")
            return

    base_dir = os.path.join(PREPROCESSED_DATA_DIR, CONSOLIDATED_PREEMBED_SUBDIR)
    os.makedirs(base_dir, exist_ok=True)

    logger.info("Loading embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model, embedding_tokenizer = load_embedding_model(EMBEDDING_MODEL_PATH, device=device)

    for company in companies:
        logger.info(f"Preembed: {company}")
        consolidated = consolidate_company_documents(
            company, manifest, PREPROCESSED_DATA_DIR, tokenizer
        )
        if not consolidated or not consolidated.chunks:
            logger.warning(f"  No content for {company}, skipping")
            continue

        company_dir = get_prebuilt_company_dir(company)
        os.makedirs(company_dir, exist_ok=True)

        save_consolidated_document_to_path(consolidated, company_dir)

        texts = [c["text"] for c in consolidated.chunks]
        embeddings = _encode_chunks_batched(
            texts,
            embedding_model,
            embedding_tokenizer,
            batch_size=EMBEDDING_BATCH_SIZE,
            max_length=EMBEDDING_MAX_LENGTH,
            device=device,
        )
        embeddings_path = os.path.join(company_dir, "_EMBEDDINGS.pt")
        torch.save(embeddings.cpu(), embeddings_path)
        logger.info(f"  Saved {len(texts)} chunk embeddings to {embeddings_path}")

    logger.info("Preembed stage complete.")
