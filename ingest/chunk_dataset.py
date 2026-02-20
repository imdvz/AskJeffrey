"""
Semantic chunking of cleaned documents.

Instead of splitting text every N characters, we use SemanticChunker to split
where the meaning shifts. This produces coherent "units of thought" chunks,
which improves retrieval quality.

How it works:
1. SemanticChunker embeds sentences using the embedding model
2. Compares adjacent sentence embeddings
3. When similarity drops significantly -> split point
4. Chunks naturally follow document structure

We also:
- Enforce a max chunk size using a fallback character splitter
- Deduplicate chunks using SHA-256
- Store metadata for traceability (doc_id, source file, chunk index)

Input:  data/cleaned.json
Output: data/chunks.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Set

from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    CLEANED_DATA_PATH,
    CHUNKS_DATA_PATH,
    EMBEDDING_MODEL_NAME,
    MAX_CHUNK_SIZE,
)


def get_sha256(text: str) -> str:
    """
    Generate a SHA-256 hash for a piece of text.
    Used to detect and skip duplicate chunks (OCR docs often repeat content).
    """
    return hashlib.sha256(text.lower().strip().encode("utf-8")).hexdigest()


def split_oversized_chunks(chunks: List[str], max_size: int) -> List[str]:
    """
    Safety net: if SemanticChunker produces a chunk that's too long,
    break it down further with a character-based splitter.
    """
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_size,
        chunk_overlap=100,  # preserves context near boundaries
    )

    result: List[str] = []
    for chunk in chunks:
        if len(chunk) > max_size:
            result.extend(fallback_splitter.split_text(chunk))
        else:
            result.append(chunk)
    return result


def main() -> None:
    # ─── Load cleaned documents ───────────────────────────
    print(f"📂 Loading cleaned documents from {CLEANED_DATA_PATH}")
    with open(CLEANED_DATA_PATH, "r", encoding="utf-8") as f:
        docs: List[Dict[str, Any]] = json.load(f)
    print(f"✅ Loaded {len(docs)} documents")

    # ─── Initialize embedding model for SemanticChunker ────
    # SemanticChunker uses embeddings internally to find meaning shift breakpoints.
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("✅ Embedding model ready")

    # ─── Initialize SemanticChunker ────────────────────────
    semantic_chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75,  # split at 75th percentile of dissimilarity
    )

    # ─── Chunk each document ───────────────────────────────
    print("✂️  Chunking documents semantically...")

    seen_hashes: Set[str] = set()
    all_chunks: List[Dict[str, Any]] = []

    for doc in tqdm(docs, desc="Chunking"):
        text = str(doc.get("text", "")).strip()
        if len(text) < 50:
            continue

        # Semantic chunking
        try:
            raw_chunks = semantic_chunker.split_text(text)
        except Exception as e:
            # Fallback: don't crash pipeline because of one bad doc
            print(f"⚠️  Semantic chunking failed for {doc.get('file', 'UNKNOWN')}: {e}")
            fallback = RecursiveCharacterTextSplitter(
                chunk_size=MAX_CHUNK_SIZE,
                chunk_overlap=100,
            )
            raw_chunks = fallback.split_text(text)

        # Enforce max size
        sized_chunks = split_oversized_chunks(raw_chunks, MAX_CHUNK_SIZE)

        # Store chunks with metadata + dedupe
        for i, chunk_text in enumerate(sized_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 30:
                continue

            chunk_hash = get_sha256(chunk_text)
            if chunk_hash in seen_hashes:
                continue
            seen_hashes.add(chunk_hash)

            all_chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "doc_id": doc.get("doc_id"),
                        "source_file": doc.get("file"),
                        "chunk_index": i,
                        "char_count": len(chunk_text),
                    },
                }
            )

    # ─── Save chunks ───────────────────────────────────────
    out_dir = os.path.dirname(CHUNKS_DATA_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(CHUNKS_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # ─── Stats ─────────────────────────────────────────────
    print(f"\n💾 Saved to {CHUNKS_DATA_PATH}")
    print(f"📊 Total chunks (after dedupe): {len(all_chunks):,}")

    # Dedupe stats: we only know how many uniques we kept (seen_hashes == uniques kept)
    # If you want “how many were skipped”, track a counter when hash repeats.
    # We'll compute skipped by tracking attempted count.
    attempted = 0
    skipped_dupes = 0
    seen_tmp: Set[str] = set()
    for c in all_chunks:
        attempted += 1
        h = get_sha256(c["text"])
        if h in seen_tmp:
            skipped_dupes += 1
        else:
            seen_tmp.add(h)
    # skipped_dupes will be 0 here since all_chunks is already deduped
    # Instead, we report unique hashes count (more meaningful)
    print(f"📊 Unique chunk hashes: {len(seen_hashes):,}")

    char_counts = [c["metadata"]["char_count"] for c in all_chunks]
    if char_counts:
        avg_chars = sum(char_counts) // len(char_counts)
        print(f"📊 Avg chunk size: {avg_chars:,} chars")
        print(f"📊 Smallest chunk: {min(char_counts):,} chars")
        print(f"📊 Largest chunk: {max(char_counts):,} chars")
    else:
        print("⚠️ No chunks produced. Check CLEANED_DATA_PATH and MAX_CHUNK_SIZE.")


if __name__ == "__main__":
    main()
