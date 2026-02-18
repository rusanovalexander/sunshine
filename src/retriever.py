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


def create_retriever_from_chunks(chunks: List[Dict]) -> BM25Retriever:
    """Create a BM25 retriever from chunk data."""
    retriever = BM25Retriever()
    texts = [chunk.get('text', '') for chunk in chunks]
    retriever.index(texts)
    return retriever


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
