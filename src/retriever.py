"""
=================================================
Retriever: BM25-based Chunk Retrieval
=================================================
Finds the most relevant chunks for each field group
without requiring external embedding models.

Enhanced with financial synonym expansion for better
recall on domain-specific terminology.
"""

import re
import math
from collections import Counter
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)


# =====================================================================
# FINANCIAL SYNONYM DICTIONARY
# =====================================================================
# Maps canonical terms to all known synonyms in project finance docs.
# BM25 expands query tokens using this dictionary so that searching
# for "spread" also matches chunks containing "margin", etc.
# This runs entirely on CPU — zero GPU cost.

FINANCIAL_SYNONYMS: Dict[str, List[str]] = {
    # Pricing & Interest
    "spread": ["margin", "applicable margin", "credit spread", "premium"],
    "margin": ["spread", "applicable margin", "credit spread", "premium"],
    "euribor": ["base rate", "reference rate", "benchmark", "screen rate"],
    "sofr": ["base rate", "reference rate", "benchmark", "secured overnight"],
    "libor": ["base rate", "reference rate", "benchmark", "screen rate"],
    "sonia": ["base rate", "reference rate", "benchmark"],
    "rate": ["interest", "coupon", "yield"],
    "basis points": ["bps", "bp"],
    "bps": ["basis points", "bp"],
    "floor": ["minimum rate", "zero floor", "rate floor"],
    "cap": ["maximum rate", "rate cap", "ceiling"],

    # Facility & Structure
    "facility": ["credit facility", "loan", "financing", "credit line", "accommodation"],
    "tranche": ["facility", "sub facility", "portion", "slice"],
    "commitment": ["aggregate commitment", "total commitment", "facility amount", "credit limit"],
    "limit": ["commitment", "aggregate", "facility amount", "maximum amount"],
    "revolving": ["rcf", "revolver", "revolving credit", "revolving facility"],
    "term loan": ["tl", "term facility", "amortising loan", "term debt"],
    "bond": ["notes", "fixed income", "debt securities"],
    "letter of credit": ["lc", "loc", "standby", "documentary credit"],

    # Parties
    "borrower": ["obligor", "debtor", "company", "issuer", "facility agent"],
    "lender": ["creditor", "bank", "financier", "participant", "arranger"],
    "sponsor": ["equity investor", "shareholder", "promoter", "equity holder"],
    "guarantor": ["surety", "guarantee provider", "credit support"],
    "contractor": ["epc", "epc contractor", "builder", "construction company", "d&c"],

    # Dates & Schedules
    "maturity": ["termination date", "final repayment", "expiry", "final maturity date", "tenor"],
    "inception": ["signing date", "effective date", "commencement", "closing date", "dated"],
    "drawdown": ["utilisation", "utilization", "disbursement", "advance"],
    "repayment": ["amortisation", "amortization", "principal payment", "instalment", "installment"],
    "schedule": ["timetable", "profile", "amortisation schedule", "repayment profile"],
    "tenor": ["term", "duration", "maturity", "life"],

    # Covenants & Ratios
    "covenant": ["financial covenant", "undertaking", "condition", "restriction"],
    "dscr": ["debt service coverage", "debt service coverage ratio", "ds coverage"],
    "llcr": ["loan life coverage", "loan life coverage ratio"],
    "icr": ["interest coverage", "interest coverage ratio", "interest cover"],
    "ebitda": ["earnings", "operating profit", "cash flow"],
    "leverage": ["gearing", "debt to equity", "debt ratio", "net debt"],
    "lock-up": ["lockup", "lock up", "dividend stopper", "distribution restriction", "dividend trap"],
    "dividend": ["distribution", "shareholder payment", "equity distribution"],
    "reserve": ["reserve account", "dsra", "debt service reserve", "cash reserve"],

    # Hedging
    "hedge": ["hedging", "swap", "derivative", "risk mitigation"],
    "swap": ["interest rate swap", "irs", "derivative", "hedge"],
    "notional": ["notional amount", "nominal", "face value", "principal amount"],
    "floating": ["variable", "adjustable", "index linked"],
    "fixed": ["fixed rate", "fixed coupon", "fixed interest"],

    # Cash Sweep
    "cash sweep": ["mandatory prepayment", "excess cash", "cash waterfall", "sweep mechanism"],
    "prepayment": ["early repayment", "voluntary prepayment", "mandatory prepayment", "sweep"],
    "waterfall": ["cash waterfall", "payment waterfall", "priority of payments"],

    # Fees
    "upfront fee": ["arrangement fee", "structuring fee", "front end fee", "closing fee"],
    "commitment fee": ["undrawn fee", "availability fee", "non utilisation fee", "non utilization fee"],
    "agency fee": ["agent fee", "facility agent fee", "administrative fee"],

    # Revenue & Offtake
    "offtake": ["ppa", "power purchase agreement", "off take", "purchase agreement"],
    "ppa": ["power purchase agreement", "offtake agreement", "energy purchase"],
    "tariff": ["regulated tariff", "feed in tariff", "fit", "regulated price"],
    "revenue": ["income", "cash flow", "receipts", "proceeds"],

    # Syndication
    "syndicate": ["syndication", "club deal", "participation"],
    "syndication": ["syndicate", "club deal", "participation", "co-lending"],
    "ing": ["ing bank", "ing group", "internationale nederlanden"],
    "share": ["participation", "proportion", "percentage", "commitment share"],

    # Project
    "project": ["asset", "venture", "development", "scheme"],
    "construction": ["build", "development", "erection", "installation"],
    "operation": ["operating", "operational", "commercial operation", "cod"],
    "completion": ["completion date", "cod", "commercial operation date", "project completion"],
    "sector": ["industry", "segment", "nace", "classification"],
    "location": ["country", "jurisdiction", "geography", "situs", "situated"],
    "holdco": ["holding company", "hold co", "parent company", "spv parent"],
}

# Build reverse lookup: token -> set of all synonyms (including self)
_SYNONYM_LOOKUP: Dict[str, Set[str]] = {}

def _build_synonym_lookup():
    """Build a fast token-level synonym lookup from FINANCIAL_SYNONYMS."""
    for canonical, synonyms in FINANCIAL_SYNONYMS.items():
        # Tokenize canonical and each synonym into individual words
        all_terms = [canonical] + synonyms
        all_tokens = set()
        for term in all_terms:
            for token in re.findall(r'\b[a-z0-9]+\b', term.lower()):
                all_tokens.add(token)
        # Each token maps to all tokens in the group
        for token in all_tokens:
            if token not in _SYNONYM_LOOKUP:
                _SYNONYM_LOOKUP[token] = set()
            _SYNONYM_LOOKUP[token].update(all_tokens)

_build_synonym_lookup()


def expand_with_synonyms(tokens: List[str], max_expansion: int = 3) -> List[str]:
    """
    Expand a list of query tokens with financial synonyms.

    For each token, adds up to max_expansion synonym tokens.
    This ensures BM25 matches chunks that use alternative terminology.
    """
    expanded = list(tokens)  # Keep originals first
    seen = set(tokens)

    for token in tokens:
        if token in _SYNONYM_LOOKUP:
            synonyms = _SYNONYM_LOOKUP[token] - seen
            added = 0
            for syn in synonyms:
                if added >= max_expansion:
                    break
                expanded.append(syn)
                seen.add(syn)
                added += 1

    return expanded


class BM25Retriever:
    """
    BM25 retrieval for finding relevant document chunks.
    Works entirely locally - no external dependencies.

    Enhanced with financial synonym expansion: query tokens are
    expanded with domain-specific synonyms before scoring, so
    searching for "spread" also scores chunks containing "margin".
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, use_synonyms: bool = True):
        self.k1 = k1
        self.b = b
        self.use_synonyms = use_synonyms
        self.documents = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}  # term -> number of docs containing term
        self.idf = {}
        self.doc_term_freqs = []  # list of Counter objects

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens

    def index(self, documents: List[str]):
        """Index a list of documents."""
        self.documents = documents
        self.doc_term_freqs = []
        self.doc_lengths = []
        term_doc_count = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)

            # Count documents containing each term
            for term in set(tokens):
                term_doc_count[term] += 1

        self.avg_doc_length = sum(self.doc_lengths) / len(documents) if documents else 0

        # Calculate IDF for each term
        n_docs = len(documents)
        for term, doc_count in term_doc_count.items():
            self.idf[term] = math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]

        for term in query_tokens:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self.idf.get(term, 0)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """
        Search for documents matching query.
        Returns list of (doc_idx, score, text) tuples.

        When use_synonyms is True, query tokens are expanded with
        financial domain synonyms before scoring.
        """
        query_tokens = self._tokenize(query)

        if self.use_synonyms:
            query_tokens = expand_with_synonyms(query_tokens)

        scores = []
        for idx in range(len(self.documents)):
            score = self._score(query_tokens, idx)
            if score > 0:
                scores.append((idx, score, self.documents[idx]))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def search_with_keywords(self, keywords: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Search using a list of keywords (combined query)."""
        query = ' '.join(keywords)
        return self.search(query, top_k)


# =====================================================================
# EMBEDDING-BASED RETRIEVER (Qwen3-Embedding-0.6B)
# =====================================================================

class EmbeddingRetriever:
    """
    Dense embedding retriever using Qwen3-Embedding-0.6B.

    Encodes all document chunks into dense vectors during index(),
    then retrieves via cosine similarity during search().

    The model is loaded in fp16 (~1.2GB VRAM) and kept on GPU
    alongside the main Qwen3-14B LLM.
    """

    def __init__(self, model=None, tokenizer=None, batch_size: int = 32,
                 device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.documents = []
        self.doc_embeddings = None  # (num_docs, embed_dim) tensor

    def index(self, documents: List[str]):
        """Encode all documents into embeddings."""
        import torch

        if self.model is None:
            raise RuntimeError("EmbeddingRetriever requires a loaded embedding model. "
                               "Pass model and tokenizer to constructor.")

        self.documents = documents

        if not documents:
            self.doc_embeddings = None
            return

        logger.info(f"  Encoding {len(documents)} chunks with embedding model...")

        all_embeddings = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            embs = self._encode(batch)
            all_embeddings.append(embs)

        # Concatenate and normalize
        self.doc_embeddings = torch.cat(all_embeddings, dim=0)
        self.doc_embeddings = torch.nn.functional.normalize(self.doc_embeddings, p=2, dim=1)

        # Store on CPU to save GPU memory; move to GPU only during search
        self.doc_embeddings = self.doc_embeddings.cpu()

        logger.info(f"  Indexed {len(documents)} chunks → embeddings shape {self.doc_embeddings.shape}")

    def _encode(self, texts: List[str]):
        """Encode a batch of texts into embeddings."""
        import torch

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use last hidden state with mean pooling (standard for embedding models)
        attention_mask = inputs['attention_mask']
        hidden = outputs.last_hidden_state
        # Masked mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        return embeddings

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Search for documents matching query via cosine similarity."""
        import torch

        if self.doc_embeddings is None or len(self.documents) == 0:
            return []

        # Encode query (with instruction prefix for Qwen3-Embedding)
        query_with_instruction = f"Instruct: Find relevant financial document sections\nQuery: {query}"
        query_emb = self._encode([query_with_instruction])
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)

        # Cosine similarity (query is on GPU, docs are on CPU — move docs temporarily)
        doc_emb_device = self.doc_embeddings.to(query_emb.device)
        scores = torch.mm(query_emb, doc_emb_device.t()).squeeze(0)  # (num_docs,)

        # Get top-k
        k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(scores, k)

        results = []
        for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
            if score > 0:
                results.append((idx, score, self.documents[idx]))

        return results

    def search_with_keywords(self, keywords: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Search using a list of keywords (combined query)."""
        query = ' '.join(keywords)
        return self.search(query, top_k)


# =====================================================================
# HYBRID RETRIEVER (BM25 + Embedding)
# =====================================================================

class HybridRetriever:
    """
    Combines BM25 and Embedding retriever scores.

    Normalizes scores from each retriever to [0, 1] using min-max,
    then combines with configurable weight.
    """

    def __init__(self, bm25_retriever: BM25Retriever,
                 embedding_retriever: EmbeddingRetriever,
                 bm25_weight: float = 0.5):
        self.bm25 = bm25_retriever
        self.embedding = embedding_retriever
        self.bm25_weight = bm25_weight
        self.embedding_weight = 1.0 - bm25_weight
        self.documents = []

    def index(self, documents: List[str]):
        """Index documents in both retrievers."""
        self.documents = documents
        self.bm25.index(documents)
        self.embedding.index(documents)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Search using both retrievers and combine scores."""
        # Get results from both (fetch more than top_k for better merging)
        fetch_k = min(top_k * 3, len(self.documents)) if self.documents else top_k
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        emb_results = self.embedding.search(query, top_k=fetch_k)

        # Build score maps: doc_idx -> score
        bm25_scores = {idx: score for idx, score, _ in bm25_results}
        emb_scores = {idx: score for idx, score, _ in emb_results}

        # Normalize each to [0, 1]
        bm25_norm = self._normalize_scores(bm25_scores)
        emb_norm = self._normalize_scores(emb_scores)

        # Combine scores for all documents seen by either retriever
        all_indices = set(bm25_norm.keys()) | set(emb_norm.keys())
        combined = {}
        for idx in all_indices:
            b_score = bm25_norm.get(idx, 0.0)
            e_score = emb_norm.get(idx, 0.0)
            combined[idx] = self.bm25_weight * b_score + self.embedding_weight * e_score

        # Sort by combined score
        sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in sorted_results[:top_k]:
            if score > 0:
                results.append((idx, score, self.documents[idx]))

        return results

    def search_with_keywords(self, keywords: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Search using a list of keywords (combined query)."""
        query = ' '.join(keywords)
        return self.search(query, top_k)

    @staticmethod
    def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalize scores to [0, 1]."""
        if not scores:
            return {}
        values = list(scores.values())
        min_s = min(values)
        max_s = max(values)
        if max_s == min_s:
            return {k: 1.0 for k in scores}
        return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}


# =====================================================================
# EMBEDDING MODEL LOADING
# =====================================================================

def load_embedding_model(model_path: str, device: str = "cuda"):
    """
    Load Qwen3-Embedding-0.6B for dense retrieval.

    ~1.2GB in fp16 — fits alongside Qwen3-14B 4-bit on A100 20GB.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"Loading embedding model from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    logger.info(f"  Embedding model loaded on {device}")
    return model, tokenizer


# =====================================================================
# FACTORY FUNCTION
# =====================================================================

def create_retriever_from_chunks(
    chunks: List[Dict],
    retriever_type: str = "bm25",
    embedding_model=None,
    embedding_tokenizer=None,
    bm25_weight: float = 0.5
):
    """
    Create a retriever from chunk data.

    Args:
        chunks: List of chunk dicts with 'text' key
        retriever_type: "bm25", "embedding", or "hybrid"
        embedding_model: Pre-loaded embedding model (required for embedding/hybrid)
        embedding_tokenizer: Pre-loaded embedding tokenizer
        bm25_weight: Weight for BM25 in hybrid mode (default 0.5)

    Returns:
        Initialized retriever with same .search() / .search_with_keywords() interface
    """
    texts = [chunk.get('text', '') for chunk in chunks]

    if retriever_type == "bm25":
        retriever = BM25Retriever()
        retriever.index(texts)
        return retriever

    elif retriever_type == "embedding":
        if embedding_model is None:
            raise ValueError("embedding_model is required for 'embedding' retriever type. "
                             "Load it with load_embedding_model() first.")
        retriever = EmbeddingRetriever(
            model=embedding_model,
            tokenizer=embedding_tokenizer,
        )
        retriever.index(texts)
        return retriever

    elif retriever_type == "hybrid":
        if embedding_model is None:
            raise ValueError("embedding_model is required for 'hybrid' retriever type. "
                             "Load it with load_embedding_model() first.")
        bm25 = BM25Retriever()
        emb = EmbeddingRetriever(
            model=embedding_model,
            tokenizer=embedding_tokenizer,
        )
        retriever = HybridRetriever(bm25, emb, bm25_weight=bm25_weight)
        retriever.index(texts)
        return retriever

    else:
        raise ValueError(f"Unknown retriever_type: {retriever_type}. "
                         f"Options: 'bm25', 'embedding', 'hybrid'")


def retrieve_for_field_group(retriever: BM25Retriever, 
                             keywords: List[str], 
                             field_names: List[str],
                             top_k: int = 8) -> List[str]:
    """
    Retrieve relevant chunks for a field group.
    Uses both group keywords and field names as search terms.
    """
    # Combine keywords with field name terms
    search_terms = keywords.copy()
    for field_name in field_names:
        # Extract meaningful words from field names
        words = re.findall(r'\b[a-z]+\b', field_name.lower())
        search_terms.extend(words)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in search_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    results = retriever.search_with_keywords(unique_terms, top_k)
    return [text for _, _, text in results]


def retrieve_for_specific_field(retriever: BM25Retriever,
                                field_name: str,
                                field_description: str,
                                additional_keywords: List[str] = None,
                                top_k: int = 5) -> List[str]:
    """Retrieve chunks specifically relevant to a single field."""
    # Build search query from field info
    query_parts = [field_name, field_description]
    if additional_keywords:
        query_parts.extend(additional_keywords)
    
    query = ' '.join(query_parts)
    results = retriever.search(query, top_k)
    return [text for _, _, text in results]
