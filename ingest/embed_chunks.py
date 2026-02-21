"""
Embed chunks into ChromaDB + build BM25 keyword index.

This step builds two indexes:

1) DENSE INDEX (ChromaDB)
   - Embed each chunk using the embedding model
   - Store vectors in a persistent Chroma collection
   - Used for semantic search

2) SPARSE INDEX (BM25)
   - Build BM25 keyword index over chunk texts
   - Save BM25 object + chunk mapping to disk
   - Used for keyword matching (names, dates, identifiers)

Input:  data/chunks.json
Output: chroma_db/ (vector store), data/bm25_index.pkl, data/bm25_chunks.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    CHUNKS_DATA_PATH,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBED_BATCH_SIZE,
    BM25_INDEX_PATH,
    BM25_CHUNKS_PATH,
)


def ensure_dir_for_file(path: str) -> None:
    """Create parent directory for a file path if it doesn't exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def chunk_to_document_and_id(chunk: Dict[str, Any]) -> Tuple[Document, str]:
    """Convert a chunk dict into a LangChain Document + deterministic chunk ID."""
    meta = chunk.get("metadata", {}) or {}
    doc_id = meta.get("doc_id", "unknown_doc")
    chunk_index = meta.get("chunk_index", "0")

    # Deterministic ID prevents duplicates on rerun
    chunk_uid = f"{doc_id}_{chunk_index}"

    doc = Document(
        page_content=str(chunk.get("text", "")),
        metadata=meta,
    )
    return doc, chunk_uid


def build_chroma_index(chunks: List[Dict[str, Any]], embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Embed all chunks and store them in ChromaDB (persistent).

    Uses deterministic IDs so re-running won't create duplicates.
    """
    print("\n🔮 Building ChromaDB vector index...")
    print(f"   Model: {EMBEDDING_MODEL_NAME}")
    print(f"   Batch size: {EMBED_BATCH_SIZE}")
    print(f"   Output dir: {CHROMA_DIR}/")
    print(f"   Collection: {CHROMA_COLLECTION_NAME}")

    ensure_dir(CHROMA_DIR)

    # Create / load persistent Chroma collection once
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
    )

    # Convert to documents + ids (streamed in batches)
    total = len(chunks)
    if total == 0:
        print("⚠️ No chunks found. Skipping Chroma build.")
        return vectorstore

    # Add in batches
    for start in tqdm(range(0, total, EMBED_BATCH_SIZE), desc="Embedding batches"):
        batch = chunks[start : start + EMBED_BATCH_SIZE]

        docs: List[Document] = []
        ids: List[str] = []
        for ch in batch:
            doc, doc_id = chunk_to_document_and_id(ch)
            # skip empty
            if not doc.page_content.strip():
                continue
            docs.append(doc)
            ids.append(doc_id)

        if docs:
            vectorstore.add_documents(documents=docs, ids=ids)

    print(f"✅ ChromaDB index updated: attempted {total:,} chunks")
    return vectorstore


def build_bm25_index(chunks: List[Dict[str, Any]]) -> None:
    """
    Build a BM25 keyword index for sparse retrieval and save it.

    Tokenization: lowercase + whitespace split (fast, ok for English OCR-ish text).
    """
    print("\n📚 Building BM25 keyword index...")

    tokenized_corpus: List[List[str]] = []
    bm25_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        tokenized_corpus.append(text.lower().split())
        bm25_chunks.append(
            {
                "text": text,
                "metadata": chunk.get("metadata", {}) or {},
            }
        )

    bm25_index = BM25Okapi(tokenized_corpus)

    ensure_dir_for_file(BM25_INDEX_PATH)
    ensure_dir_for_file(BM25_CHUNKS_PATH)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_index, f)

    with open(BM25_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(bm25_chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ BM25 index built: {len(tokenized_corpus):,} chunks indexed")
    print(f"💾 Saved: {BM25_INDEX_PATH}")
    print(f"💾 Saved: {BM25_CHUNKS_PATH}")


def main() -> None:
    # ─── Load chunks ──────────────────────────────────────
    print(f"📂 Loading chunks from {CHUNKS_DATA_PATH}")
    with open(CHUNKS_DATA_PATH, "r", encoding="utf-8") as f:
        chunks: List[Dict[str, Any]] = json.load(f)
    print(f"✅ Loaded {len(chunks):,} chunks")

    # ─── Load embedding model ─────────────────────────────
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("✅ Embedding model ready")

    # ─── Build indexes ────────────────────────────────────
    build_chroma_index(chunks, embeddings)
    build_bm25_index(chunks)

    print("\n🎉 Ingestion complete! Both indexes are ready.")
    print(f"   📁 ChromaDB dir: {CHROMA_DIR}/")
    print(f"   📁 BM25 index:  {BM25_INDEX_PATH}")
    print(f"   📁 BM25 chunks: {BM25_CHUNKS_PATH}")


if __name__ == "__main__":
    main()
