"""
Hybrid Retriever — combines dense (vector) and sparse (keyword) search.

Why hybrid? Neither search method is perfect alone:
- Dense (ChromaDB)  → great at MEANING ("Caribbean property" finds "Epstein's island")
                    → bad at exact matches ("document JE-1045")
- Sparse (BM25)     → great at EXACT WORDS ("Maxwell", "2005", "N908JE")
                    → bad at understanding meaning or synonyms

Hybrid search runs BOTH, then merges results using Reciprocal Rank Fusion (RRF).

RRF — How it works:
  Each result gets a score based on its RANK (position) in each list, not its raw score.
  Formula: RRF_score = 1 / (rank + k)   where k=60 (a smoothing constant)

Input:  User's query string
Output: Top-K merged chunks ready for re-ranking

Dependencies: chroma_db/ and data/bm25_index.pkl from Step 9
"""

import json
import pickle
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    BM25_INDEX_PATH,
    BM25_CHUNKS_PATH,
    VECTOR_FETCH_K,
    BM25_FETCH_K,
    HYBRID_TOP_K,
    RRF_K,

    # new config knobs
    ENABLE_ISLAND_QUERY_EXPANSION,
    ISLAND_EXPANSION_TERMS,
    ISLAND_INTENT_KEYWORDS,
    ISLAND_INTENT_VERBS,

    ENABLE_IDENTIFIER_BOOST,
    IDENTIFIER_PATTERNS,
    DENSE_WEIGHT_DEFAULT,
    SPARSE_WEIGHT_DEFAULT,
    DENSE_WEIGHT_IDENTIFIER,
    SPARSE_WEIGHT_IDENTIFIER,
)


def _normalize(s: str) -> str:
    return (s or "").lower().strip()


# Compile identifier patterns once
_IDENTIFIER_REGEXES = [re.compile(p) for p in IDENTIFIER_PATTERNS]


def is_identifier_query(query: str) -> bool:
    """
    Heuristic: if query matches common identifier formats (tail numbers, doc IDs, exhibit IDs),
    prefer BM25 dominance.
    """
    q = query or ""
    for rx in _IDENTIFIER_REGEXES:
        if rx.search(q):
            return True
    return False


def is_island_intent(query: str) -> bool:
    """
    Only expand BM25 query when island intent is implied.
    Rules:
    - If any island keyword appears -> True
    - Else if 'epstein' appears AND any travel/visit verb appears -> True
    """
    q = _normalize(query)

    # Direct island keyword triggers
    for kw in ISLAND_INTENT_KEYWORDS:
        if kw in q:
            return True

    # Implied island intent: Epstein + visit/travel verbs
    if "epstein" in q:
        for v in ISLAND_INTENT_VERBS:
            if v in q:
                return True

    return False


def expand_query_for_bm25(query: str) -> str:
    """
    Append island expansion terms ONLY when island intent is implied.
    """
    if not ENABLE_ISLAND_QUERY_EXPANSION:
        return query

    if is_island_intent(query):
        extra = " ".join(ISLAND_EXPANSION_TERMS)
        return f"{query} {extra}"
    return query


class HybridRetriever:
    """
    Combines ChromaDB (dense/semantic) and BM25 (sparse/keyword) retrieval
    into a single unified search interface using Reciprocal Rank Fusion.
    """

    def __init__(self):
        """
        Initialize both search indexes. This loads everything into memory,
        so it takes a few seconds on first call — but subsequent queries are fast.
        """
        # ─── Dense retriever (ChromaDB) ───────────────────
        print("🔮 Loading ChromaDB vector store...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
        )
        print(f"✅ ChromaDB loaded: {self.vectorstore._collection.count():,} vectors")

        # ─── Sparse retriever (BM25) ─────────────────────
        print("📚 Loading BM25 index...")
        with open(BM25_INDEX_PATH, 'rb') as f:
            self.bm25_index: BM25Okapi = pickle.load(f)

        with open(BM25_CHUNKS_PATH, 'r', encoding='utf-8') as f:
            self.bm25_chunks: list = json.load(f)

        print(f"✅ BM25 loaded: {len(self.bm25_chunks):,} chunks indexed")

    def _search_dense(self, query: str, k: int = VECTOR_FETCH_K) -> list[dict]:
        """
        Semantic search via ChromaDB.

        Embeds the query and finds the K most similar chunk vectors.
        Returns chunks ranked by cosine similarity (most similar first).
        """
        results = self.vectorstore.similarity_search(query, k=k)

        return [
            {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'source': 'dense',
            }
            for doc in results
        ]

    def _search_sparse(self, query: str, k: int = BM25_FETCH_K) -> list[dict]:
        """
        Keyword search via BM25.

        Tokenizes the query and scores every chunk based on term frequency
        and inverse document frequency. Returns the top K scoring chunks.
        """
        # Expand query ONLY when island intent is implied
        expanded_query = expand_query_for_bm25(query)

        # Tokenize the query the same way we tokenized the corpus (lowercase + split)
        query_tokens = expanded_query.lower().split()

        # Get BM25 scores for all chunks
        scores = self.bm25_index.get_scores(query_tokens)

        # Get the indices of the top K scores (sorted highest first)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            {
                'text': self.bm25_chunks[i]['text'],
                'metadata': self.bm25_chunks[i]['metadata'],
                'source': 'sparse',
                'bm25_score': float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0  # skip chunks with zero relevance
        ]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        k: int = RRF_K,
        dense_weight: float = DENSE_WEIGHT_DEFAULT,
        sparse_weight: float = SPARSE_WEIGHT_DEFAULT,
    ) -> list[dict]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion (RRF),
        with optional weighting.

        RRF uses rank-based scoring to avoid mixing incompatible raw score scales.
        We optionally weight dense vs sparse contributions.
        """
        fused_scores = {}  # text → cumulative RRF score
        chunk_map = {}     # text → chunk data

        # Score dense results by their rank
        for rank, result in enumerate(dense_results):
            text = result['text']
            rrf_score = dense_weight * (1.0 / (rank + k))
            fused_scores[text] = fused_scores.get(text, 0) + rrf_score
            chunk_map[text] = result

        # Score sparse results by their rank
        for rank, result in enumerate(sparse_results):
            text = result['text']
            rrf_score = sparse_weight * (1.0 / (rank + k))
            fused_scores[text] = fused_scores.get(text, 0) + rrf_score
            if text not in chunk_map:
                chunk_map[text] = result

        # Sort by fused RRF score (highest first)
        sorted_texts = sorted(fused_scores.keys(), key=lambda t: fused_scores[t], reverse=True)

        fused_results = []
        for text in sorted_texts:
            chunk = chunk_map[text].copy()
            chunk['rrf_score'] = fused_scores[text]
            chunk['source'] = 'hybrid'
            fused_results.append(chunk)

        return fused_results

    def retrieve(self, query: str, top_k: int = HYBRID_TOP_K) -> list[dict]:
        """
        Main entry point: run hybrid search and return the top K merged results.

        Flow:
        1. Run dense search (ChromaDB)
        2. Run sparse search (BM25) with optional island query expansion
        3. Merge with weighted RRF (BM25 dominates for identifier-like queries)
        """
        # Step 1: Get results from both search methods
        dense_results = self._search_dense(query)
        sparse_results = self._search_sparse(query)

        # Step 2: Decide weights (identifier queries -> BM25 dominates)
        dense_w = DENSE_WEIGHT_DEFAULT
        sparse_w = SPARSE_WEIGHT_DEFAULT

        if ENABLE_IDENTIFIER_BOOST and is_identifier_query(query):
            dense_w = DENSE_WEIGHT_IDENTIFIER
            sparse_w = SPARSE_WEIGHT_IDENTIFIER

        # Step 3: Merge using weighted RRF
        fused_results = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            dense_weight=dense_w,
            sparse_weight=sparse_w,
        )

        # Step 4: Return the top K
        return fused_results[:top_k]


# ──────────────────────────────────────────────────────
# Quick test — run this file directly to verify it works
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    retriever = HybridRetriever()

    test_queries = [
        "Who visited Epstein's island?",
        "flight log N908JE",
        "Maxwell deposition testimony",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=5)

        for i, r in enumerate(results):
            print(f"\n--- Result {i+1} (RRF: {r['rrf_score']:.4f}) [{r['metadata']['source_file']}] ---")
            print(r['text'][:200])