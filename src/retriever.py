"""
=================================================
Retriever: BM25-based Chunk Retrieval
=================================================
Finds the most relevant chunks for each field group
without requiring external embedding models.
"""

import re
import math
from collections import Counter
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    BM25 retrieval for finding relevant document chunks.
    Works entirely locally - no external dependencies.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
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
        """
        query_tokens = self._tokenize(query)
        
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
