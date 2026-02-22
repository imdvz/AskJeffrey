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

  Example:
    Chunk A: Rank 1 in ChromaDB, Rank 5 in BM25
      → RRF = 1/(1+60) + 1/(5+60) = 0.0164 + 0.0154 = 0.0318

    Chunk B: Rank 3 in ChromaDB, Rank 2 in BM25
      → RRF = 1/(3+60) + 1/(2+60) = 0.0159 + 0.0161 = 0.0320

    Chunk B wins! It ranked well in BOTH lists, making it more likely to be relevant.

  Why ranks instead of raw scores?
    - ChromaDB scores and BM25 scores are on completely different scales
    - You can't directly compare 0.87 (cosine similarity) with 23.5 (BM25 score)
    - Ranks normalize everything to a fair playing field

Input:  User's query string
Output: Top-K merged chunks ready for re-ranking

Dependencies: chroma_db/ and data/bm25_index.pkl from Step 9
"""

import json
import os
import pickle
import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    BM25_CHUNKS_PATH,
    BM25_FETCH_K,
    BM25_INDEX_PATH,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME,
    HYBRID_TOP_K,
    RRF_K,
    VECTOR_FETCH_K,
)


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
        with open(BM25_INDEX_PATH, "rb") as f:
            self.bm25_index: BM25Okapi = pickle.load(f)

        with open(BM25_CHUNKS_PATH, "r", encoding="utf-8") as f:
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
                "text": doc.page_content,
                "metadata": doc.metadata,
                "source": "dense",
            }
            for doc in results
        ]

    def _search_sparse(self, query: str, k: int = BM25_FETCH_K) -> list[dict]:
        """
        Keyword search via BM25.

        Tokenizes the query and scores every chunk based on term frequency
        and inverse document frequency. Returns the top K scoring chunks.
        """
        # Tokenize the query the same way we tokenized the corpus (lowercase + split)
        query_tokens = query.lower().split()

        # Get BM25 scores for all chunks
        scores = self.bm25_index.get_scores(query_tokens)

        # Get the indices of the top K scores (sorted highest first)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            {
                "text": self.bm25_chunks[i]["text"],
                "metadata": self.bm25_chunks[i]["metadata"],
                "source": "sparse",
                "bm25_score": float(scores[i]),
            }
            for i in top_indices
            if scores[i] > 0  # skip chunks with zero relevance
        ]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        k: int = RRF_K,
    ) -> list[dict]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion (RRF).

        Instead of comparing raw scores (which are on different scales),
        RRF uses the RANK of each result. A chunk that appears high in
        both lists gets a higher combined score than one that's high in
        only one list.

        The 'k' parameter (default 60) controls how much we favor top-ranked
        results vs. lower-ranked ones. Higher k = more equal weighting.
        """
        # We use the chunk text as the unique key for merging
        # (two results with the same text are the same chunk)
        fused_scores = {}  # text → cumulative RRF score
        chunk_map = {}  # text → chunk data (for returning later)

        # Score dense results by their rank
        for rank, result in enumerate(dense_results):
            text = result["text"]
            rrf_score = 1.0 / (rank + k)
            fused_scores[text] = fused_scores.get(text, 0) + rrf_score
            chunk_map[text] = result

        # Score sparse results by their rank
        for rank, result in enumerate(sparse_results):
            text = result["text"]
            rrf_score = 1.0 / (rank + k)
            fused_scores[text] = fused_scores.get(text, 0) + rrf_score
            # If this chunk wasn't in dense results, add it to the map
            if text not in chunk_map:
                chunk_map[text] = result

        # Sort by fused RRF score (highest first)
        sorted_texts = sorted(fused_scores.keys(), key=lambda t: fused_scores[t], reverse=True)

        # Build the final list with RRF scores attached
        fused_results = []
        for text in sorted_texts:
            chunk = chunk_map[text].copy()
            chunk["rrf_score"] = fused_scores[text]
            chunk["source"] = "hybrid"
            fused_results.append(chunk)

        return fused_results

    def retrieve(self, query: str, top_k: int = HYBRID_TOP_K) -> list[dict]:
        """
        Main entry point: run hybrid search and return the top K merged results.

        Flow:
        1. Run dense search (ChromaDB) → top 40 by meaning
        2. Run sparse search (BM25)    → top 40 by keywords
        3. Merge with RRF              → top 15 best of both worlds
        """
        # Step 1: Get results from both search methods
        dense_results = self._search_dense(query)
        sparse_results = self._search_sparse(query)

        # Step 2: Merge using RRF
        fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Step 3: Return the top K
        return fused_results[:top_k]


# ──────────────────────────────────────────────────────
# Quick test — run this file directly to verify it works
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
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
            print(
                f"\n--- Result {i+1} (RRF: {r['rrf_score']:.4f}) [{r['metadata']['source_file']}] ---"
            )
            print(r["text"][:200])