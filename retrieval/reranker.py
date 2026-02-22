"""
Deduplicate + Cross-Encoder Reranking.

After hybrid retrieval (Step 10), we have ~15 chunks ranked by RRF. But RRF
ranking is based on POSITION in two lists — it doesn't actually understand
whether a chunk answers the user's question.

This is where the cross-encoder reranker comes in.

Why a cross-encoder?
  - Bi-encoders (like our BGE embedding model) encode query and chunk SEPARATELY,
    then compare vectors. Fast, but shallow understanding.
  - Cross-encoders encode query AND chunk TOGETHER as a single input, allowing
    deep token-level attention between them. Much more accurate, but slower.

  That's why we use bi-encoders for the initial broad search (40+40 chunks)
  and cross-encoders only for the final shortlist (15 chunks). Best of both worlds.

Before reranking, we also deduplicate chunks that are near-identical.
RRF merges by exact text match, but OCR'd documents often produce chunks that are
~90% similar (e.g., same paragraph with minor OCR differences). We catch those here.

Pipeline:
  15 chunks from RRF → Deduplicate → Cross-Encoder Rerank → Top 6 → LLM

Input:  List of hybrid-retrieved chunks + the user's query
Output: Top N most relevant chunks, scored by the cross-encoder

Dependencies: cross-encoder/ms-marco-MiniLM-L-6-v2 (downloaded automatically on first run)
"""

from sentence_transformers import CrossEncoder
from difflib import SequenceMatcher

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANKER_MODEL_NAME, RERANK_TOP_N

# ──────────────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────────────

def _text_similarity(a: str, b: str) -> float:
    """
    Compute text similarity between two strings using SequenceMatcher.

    Returns a float between 0.0 (completely different) and 1.0 (identical).

    We use this instead of exact matching because OCR'd documents frequently
    produce near-duplicate chunks — same paragraph with minor differences like:
      - "Jeffrey Epstein" vs "Jeffrey Epsteln" (OCR typo)
      - Extra whitespace or punctuation differences
      - A few words added/removed at chunk boundaries

    SequenceMatcher is fast enough for 15 chunks and catches these fuzzy dupes.
    """
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

def deduplicate_chunks(
    chunks: list[dict],
    similarity_threshold: float = 0.85,
) -> list[dict]:
    """
    Remove near-duplicate chunks from the candidate list.

    Two chunks are considered duplicates if their text similarity exceeds
    the threshold (default 85%). When duplicates are found, we keep the one
    with the higher RRF score (it ranked better in retrieval).

    Why 85%? Through testing:
      - 80% → too aggressive, removes chunks that are actually different
      - 90% → too lenient, lets through obvious OCR dupes
      - 85% → sweet spot for scanned legal documents

    We compare every pair (O(n²)), but n is small (15 chunks max), so it's instant.
    """
    if not chunks:
        return []

    unique_chunks = []

    for candidate in chunks:
        is_duplicate = False

        for existing in unique_chunks:
            similarity = _text_similarity(candidate['text'], existing['text'])

            if similarity >= similarity_threshold:
                # This chunk is a near-duplicate of one we already kept.
                # The existing one has a higher or equal RRF score (since we process
                # chunks in RRF-ranked order), so we skip this candidate.
                is_duplicate = True
                break

        if not is_duplicate:
            unique_chunks.append(candidate)

    removed_count = len(chunks) - len(unique_chunks)
    if removed_count > 0:
        print(f"🧹 Deduplication: removed {removed_count} near-duplicate chunk(s)")

    return unique_chunks

# ──────────────────────────────────────────────────────
# Cross-Encoder Reranking
# ──────────────────────────────────────────────────────

class Reranker:
    """
    Cross-encoder reranker that scores each (query, chunk) pair together.

    Unlike bi-encoders which encode query and chunk separately:
      Bi-encoder:    embed(query) ↔ embed(chunk)  → cosine similarity
      Cross-encoder: model(query + chunk)          → relevance score

    The cross-encoder sees both texts simultaneously, enabling deep
    token-level interaction. It can understand things like:
      - "Who visited the island?" + chunk about "guest arrivals" → HIGH score
      - "Who visited the island?" + chunk about "island geography" → LOW score

    A bi-encoder might rank both equally (both are about "island"),
    but the cross-encoder understands the question is about PEOPLE visiting.
    """

    def __init__(self):
        """
        Load the cross-encoder model. Uses ms-marco-MiniLM-L-6-v2 which is:
        - Trained on MS MARCO (a large-scale passage ranking dataset)
        - Small and fast (~22M parameters) — perfect for reranking 15 chunks
        - Specifically designed for the task of "does this passage answer this query?"

        First run downloads the model (~80MB). Subsequent runs load from cache.
        """
        print(f"⚖️  Loading cross-encoder reranker: {RERANKER_MODEL_NAME}")
        self.model = CrossEncoder(RERANKER_MODEL_NAME)
        print("✅ Reranker ready")

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int = RERANK_TOP_N,
        deduplicate: bool = True,
    ) -> list[dict]:
        """
        Main entry point: deduplicate → rerank → return top N chunks.

        Args:
            query:       The user's question
            chunks:      List of chunk dicts from HybridRetriever.retrieve()
            top_n:       How many chunks to return (default: 6)
            deduplicate: Whether to remove near-duplicates before reranking

        Returns:
            Top N chunks sorted by cross-encoder relevance score (highest first).
            Each chunk gets a 'rerank_score' field added to it.
        """
        if not chunks:
            return []

        # ─── Step 1: Deduplicate ──────────────────────────
        # Remove near-identical chunks BEFORE reranking.
        # This prevents the cross-encoder from wasting compute on duplicates,
        # and ensures diverse results in the final output.
        if deduplicate:
            chunks = deduplicate_chunks(chunks)

        # ─── Step 2: Build (query, chunk) pairs ───────────
        # The cross-encoder needs pairs of [query, passage] to score.
        pairs = [[query, chunk['text']] for chunk in chunks]

        # ─── Step 3: Score all pairs ──────────────────────
        # The model returns a relevance score for each pair.
        # Higher score = the chunk is more likely to answer the query.
        # Scores are NOT probabilities — they can be negative or > 1.
        scores = self.model.predict(pairs)

        # ─── Step 4: Attach scores and sort ───────────────
        for chunk, score in zip(chunks, scores):
            chunk['rerank_score'] = float(score)

        # Sort by rerank score (highest = most relevant)
        ranked = sorted(chunks, key=lambda c: c['rerank_score'], reverse=True)

        # ─── Step 5: Return top N ─────────────────────────
        return ranked[:top_n]

# ──────────────────────────────────────────────────────
# Quick test — run this file directly to verify it works
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    from hybrid_retriever import HybridRetriever

    # Initialize both retriever and reranker
    retriever = HybridRetriever()
    reranker = Reranker()

    test_queries = [
        "Who visited Epstein's island?",
        "flight log N909JE",
        "Maxwell deposition testimony",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")

        # Step 1: Hybrid retrieval (15 chunks)
        hybrid_results = retriever.retrieve(query)
        print(f"📦 Hybrid retrieval returned {len(hybrid_results)} chunks")

        # Step 2: Dedupe + rerank (top 6)
        reranked = reranker.rerank(query, hybrid_results)
        print(f"⚖️  Reranker selected top {len(reranked)} chunks\n")

        for i, r in enumerate(reranked):
            print(f"--- Result {i+1} ---")
            print(f"    Rerank score: {r['rerank_score']:.4f}")
            print(f"    RRF score:    {r['rrf_score']:.4f}")
            print(f"    Source file:   {r['metadata']['source_file']}")
            print(f"    Text preview:  {r['text'][:150]}...")
            print()