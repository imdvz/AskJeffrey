"""
Pydantic data models for the API layer.

These models define the exact shape of data flowing through the app:
- What the user sends (QueryRequest)
- What we return (QueryResponse)
- How we represent source chunks (SourceChunk)
- How we represent errors (ErrorResponse)

Pydantic gives us:
- Automatic validation (wrong type? missing field? → clear error)
- Serialization (Python objects ↔ JSON with zero effort)
- Self-documenting API (FastAPI auto-generates docs from these models)

These models are used by:
- api/main.py    → FastAPI request/response handling
- core/rag_chain.py → structuring the output of run_rag_query()
- app.py         → Streamlit can also use these for type safety
"""

from pydantic import BaseModel, Field
from typing import Optional

# ──────────────────────────────────────────────────────
# Request Models (what the user sends)
# ──────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    The payload a user sends when asking a question.

    - question:  The actual question about the Epstein files
    - api_key:   User's own API key (BYOK pattern). Never stored, never logged.
                 Used only for that single LLM request and then discarded.
    - provider:  Which LLM provider to use. Maps to config.PROVIDERS dict.
                 Default is "OpenRouter" (our free tier option).
    - top_k:     Optional override for how many final chunks to feed the LLM.
                 Lower = faster + cheaper, higher = more context but noisier.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask about the Epstein files.",
        examples=["Who visited Epstein's island?"],
    )

    api_key: str = Field(
        ...,
        min_length=1,
        description="Your LLM provider API key. Used for this request only, never stored.",
        examples=["sk-or-v1-abc123..."],
    )

    provider: str = Field(
        default="OpenRouter",
        description="LLM provider name. Must match a key in config.PROVIDERS.",
        examples=["OpenRouter"],
    )

    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=12,
        description="Number of reranked chunks to feed the LLM. Default uses config value (6).",
    )

# ──────────────────────────────────────────────────────
# Response Models (what we return)
# ──────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    """
    A single source chunk returned alongside the answer.

    Displayed in the "Sources" expander in the Streamlit UI so users
    can see exactly which documents the answer was based on.

    - text:             The actual chunk text the LLM read
    - source_file:      Original filename from the Epstein files dataset
    - chunk_index:      Position of this chunk within its parent document
    - relevance_score:  Cross-encoder rerank score (higher = more relevant)
                        Helps the user gauge how confident the retrieval was
    """

    text: str = Field(
        ...,
        description="The chunk text that was fed to the LLM.",
    )

    source_file: str = Field(
        ...,
        description="Original source filename from the Epstein files.",
        examples=["TEXT-001-HOUSE_OVERSIGHT_010486.txt"],
    )

    chunk_index: int = Field(
        ...,
        description="Chunk position within the parent document.",
    )

    relevance_score: float = Field(
        ...,
        description="Cross-encoder rerank score. Higher = more relevant.",
    )

class QueryResponse(BaseModel):
    """
    The complete response returned after processing a question.

    Contains the LLM's answer, the source chunks it used, and timing
    metrics so users (and developers) can see how fast each stage was.
    """

    answer: str = Field(
        ...,
        description="The LLM-generated answer based on retrieved context.",
    )

    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Source chunks the answer was based on, sorted by relevance.",
    )

    retrieval_time_ms: float = Field(
        ...,
        description="Time spent on hybrid retrieval + reranking (milliseconds).",
    )

    generation_time_ms: float = Field(
        ...,
        description="Time spent on LLM generation (milliseconds).",
    )

class ErrorResponse(BaseModel):
    """
    Structured error response for API consumers.

    error_type helps the frontend decide HOW to display the error:
    - "invalid_key"  → st.error() with instructions to check the key
    - "rate_limit"   → st.warning() with "try again in a moment"
    - "timeout"      → st.warning() with "request took too long"
    - "no_context"   → st.info() with "no relevant documents found"
    - "unknown"      → st.error() with generic message

    This beats returning raw exception strings — the UI can react intelligently.
    """

    error: str = Field(
        ...,
        description="Human-readable error message.",
        examples=["Invalid API key. Please check and try again."],
    )

    error_type: str = Field(
        ...,
        description="Machine-readable error category for frontend handling.",
        examples=["invalid_key", "rate_limit", "timeout", "no_context", "unknown"],
    )

# ──────────────────────────────────────────────────────
# Helper: Convert raw reranker output → SourceChunk list
# ──────────────────────────────────────────────────────

def chunks_to_source_list(chunks: list[dict]) -> list[SourceChunk]:
    """
    Convert the raw chunk dicts from Reranker.rerank() into a clean
    list of SourceChunk models.

    This is the bridge between the retrieval layer (which works with
    raw dicts for flexibility) and the API layer (which needs structured
    Pydantic models for validation and serialization).

    Args:
        chunks: List of chunk dicts, each with 'text', 'metadata', and 'rerank_score'.

    Returns:
        List of SourceChunk models, ready to include in a QueryResponse.
    """
    return [
        SourceChunk(
            text=chunk['text'],
            source_file=chunk['metadata'].get('source_file', 'Unknown'),
            chunk_index=chunk['metadata'].get('chunk_index', 0),
            relevance_score=round(chunk.get('rerank_score', 0.0), 4),
        )
        for chunk in chunks
    ]

# ──────────────────────────────────────────────────────
# Quick test — verify models validate correctly
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    # Test QueryRequest validation
    print("=" * 50)
    print("Testing QueryRequest")
    print("=" * 50)

    # Valid request
    req = QueryRequest(
        question="Who visited Epstein's island?",
        api_key="sk-or-v1-test123",
        provider="OpenRouter",
    )
    print(f"✅ Valid request: {req.model_dump_json(indent=2)}")

    # Test with optional top_k
    req2 = QueryRequest(
        question="Maxwell deposition",
        api_key="sk-or-v1-test456",
        top_k=4,
    )
    print(f"✅ With top_k: {req2.model_dump_json(indent=2)}")

    # Test SourceChunk
    print(f"\n{'=' * 50}")
    print("Testing SourceChunk")
    print("=" * 50)

    chunk = SourceChunk(
        text="Epstein flew to the island on March 5.",
        source_file="TEXT-001-HOUSE_OVERSIGHT_010486.txt",
        chunk_index=0,
        relevance_score=6.3013,
    )
    print(f"✅ Source chunk: {chunk.model_dump_json(indent=2)}")

    # Test QueryResponse
    print(f"\n{'=' * 50}")
    print("Testing QueryResponse")
    print("=" * 50)

    resp = QueryResponse(
        answer="According to the documents, several individuals visited...",
        sources=[chunk],
        retrieval_time_ms=245.3,
        generation_time_ms=1023.7,
    )
    print(f"✅ Full response: {resp.model_dump_json(indent=2)}")

    # Test ErrorResponse
    print(f"\n{'=' * 50}")
    print("Testing ErrorResponse")
    print("=" * 50)

    err = ErrorResponse(
        error="Invalid API key. Please check and try again.",
        error_type="invalid_key",
    )
    print(f"✅ Error response: {err.model_dump_json(indent=2)}")

    # Test chunks_to_source_list helper
    print(f"\n{'=' * 50}")
    print("Testing chunks_to_source_list")
    print("=" * 50)

    raw_chunks = [
        {
            'text': 'Some chunk text here.',
            'metadata': {'source_file': 'FILE_A.txt', 'chunk_index': 3},
            'rerank_score': 5.432,
        },
        {
            'text': 'Another chunk text.',
            'metadata': {'source_file': 'FILE_B.txt', 'chunk_index': 1},
            'rerank_score': 3.21,
        },
    ]
    sources = chunks_to_source_list(raw_chunks)
    for s in sources:
        print(f"✅ {s.source_file} (chunk {s.chunk_index}) → score: {s.relevance_score}")