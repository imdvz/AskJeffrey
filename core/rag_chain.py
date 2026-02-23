"""
The RAG Chain — the heart of AskJeffrey.

This is the file that ties EVERYTHING together:
  Retrieval (hybrid_retriever.py) → Reranking (reranker.py) → Prompts (prompts.py) → LLM

It exposes a single function: run_rag_query(question, api_key, provider)
that both app.py (Streamlit) and api/main.py (FastAPI) can import and use.

Key design decisions:
1. Retriever + Reranker are loaded ONCE at module level (they're local, no API key needed).
   This avoids reloading ~500MB of models on every request.

2. The LLM is created ON EVERY REQUEST with the user's API key.
   This is the BYOK pattern — we never store keys, each request is independent.

3. Error handling is granular — we catch specific exceptions from the LLM provider
   and return structured errors that the frontend can display intelligently.

Flow:
  User question + API key
         ↓
  HybridRetriever.retrieve() → 15 chunks (local, no key needed)
         ↓
  Reranker.rerank() → 6 best chunks (local, no key needed)
         ↓
  has_sufficient_context() → enough relevant context?
         ↓ Yes                          ↓ No
  Format prompt with context     Return "no relevant info found"
         ↓
  Create LLM with user's key (on-the-fly)
         ↓
  LLM generates answer
         ↓
  Return QueryResponse (answer + sources + timing)
"""

import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import (
    AuthenticationError,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIStatusError,
)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    PROVIDERS,
    DEFAULT_PROVIDER,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    RERANK_TOP_N,
)
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from api.prompts import (
    SYSTEM_PROMPT,
    NO_CONTEXT_RESPONSE,
    build_user_message,
    has_sufficient_context,
)
from api.models import (
    QueryResponse,
    ErrorResponse,
    chunks_to_source_list,
)

# ──────────────────────────────────────────────────────
# Module-Level Initialization (runs once on import)
# ──────────────────────────────────────────────────────
# These are heavy objects (~500MB total) that load the embedding model,
# ChromaDB vectors, BM25 index, and cross-encoder. We load them ONCE
# and reuse across all requests. No API key needed for any of this.

print("=" * 60)
print("🚀 Initializing AskJeffrey RAG pipeline...")
print("=" * 60)

_retriever = HybridRetriever()
_reranker = Reranker()

print("=" * 60)
print("✅ RAG pipeline ready — waiting for queries")
print("=" * 60)

# ──────────────────────────────────────────────────────
# LLM Factory
# ──────────────────────────────────────────────────────

def _create_llm(api_key: str, provider: str = DEFAULT_PROVIDER) -> ChatOpenAI:
    """
    Create an LLM instance on-the-fly using the user's API key.

    Why ChatOpenAI for everything (not just OpenAI)?
    Most LLM providers (Groq, OpenRouter, Together, etc.) expose an
    OpenAI-compatible API. By using ChatOpenAI with a custom base_url,
    we get a unified interface for ALL providers without needing
    separate libraries for each one.

    This function is called per-request — the LLM object is ephemeral
    and gets garbage collected after the response is sent.

    Args:
        api_key:  The user's API key (never stored, never logged)
        provider: Provider name matching a key in config.PROVIDERS

    Returns:
        A configured ChatOpenAI instance ready to generate responses.

    Raises:
        ValueError: If the provider is not found in config.PROVIDERS
    """
    if provider not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(
            f"Unknown provider '{provider}'. Available providers: {available}"
        )

    provider_config = PROVIDERS[provider]

    return ChatOpenAI(
        api_key=api_key,
        base_url=provider_config["base_url"],
        model=provider_config["model"],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

# ──────────────────────────────────────────────────────
# Main RAG Query Function
# ──────────────────────────────────────────────────────

def run_rag_query(
    question: str,
    api_key: str,
    provider: str = DEFAULT_PROVIDER,
    top_k: int | None = None,
) -> QueryResponse | ErrorResponse:
    """
    The main entry point — processes a user's question end-to-end.

    This is the ONLY function that app.py and api/main.py need to call.
    It handles the entire pipeline: retrieval → reranking → LLM generation,
    including all error handling.

    Args:
        question: The user's question about the Epstein files
        api_key:  User's LLM provider API key (BYOK)
        provider: Which provider to use (default from config)
        top_k:    Optional override for number of reranked chunks

    Returns:
        QueryResponse on success (answer + sources + timing)
        ErrorResponse on failure (structured error for frontend)
    """

    # Use config default if no override provided
    rerank_top_n = top_k if top_k is not None else RERANK_TOP_N

    # ─── Stage 1: Retrieval (local, no API key) ──────────
    # This runs entirely on the user's machine / server.
    # Hybrid search (ChromaDB + BM25) → RRF merge → ~15 candidate chunks

    try:
        retrieval_start = time.perf_counter()

        # Get hybrid search results
        hybrid_chunks = _retriever.retrieve(question)

        # Rerank to find the best chunks
        reranked_chunks = _reranker.rerank(
            query=question,
            chunks=hybrid_chunks,
            top_n=rerank_top_n,
        )

        retrieval_time_ms = round((time.perf_counter() - retrieval_start) * 1000, 1)

    except Exception as e:
        # Retrieval errors are internal — not the user's fault
        print(f"❌ Retrieval error: {e}")
        return ErrorResponse(
            error="An internal error occurred during document retrieval. Please try again.",
            error_type="unknown",
        )

    # ─── Stage 2: Context Check ──────────────────────────
    # If the reranker says "nothing is really relevant" (all negative scores),
    # we return a graceful "I don't know" instead of forcing the LLM to hallucinate.

    if not has_sufficient_context(reranked_chunks):
        return QueryResponse(
            answer=NO_CONTEXT_RESPONSE,
            sources=chunks_to_source_list(reranked_chunks),
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=0.0,
        )

    # ─── Stage 3: LLM Generation (requires API key) ─────
    # This is where the user's API key comes in. We create a throwaway
    # LLM instance, send the prompt, get the answer, and discard it.

    try:
        generation_start = time.perf_counter()

        # Create a fresh LLM with the user's key
        llm = _create_llm(api_key=api_key, provider=provider)

        # Build the messages for the LLM
        # System message: tells the LLM who it is and how to behave
        # Human message: context chunks + the user's question
        user_message = build_user_message(question, reranked_chunks)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        # Call the LLM
        response = llm.invoke(messages)

        generation_time_ms = round((time.perf_counter() - generation_start) * 1000, 1)

        # ─── Stage 4: Build response ─────────────────────
        return QueryResponse(
            answer=response.content,
            sources=chunks_to_source_list(reranked_chunks),
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
        )

    # ─── Error Handling ───────────────────────────────────
    # Each provider throws slightly different exceptions, but since we use
    # the OpenAI SDK under the hood, they all map to these error types.
    # We catch them specifically so the frontend can display the right message.

    except AuthenticationError:
        # The API key is invalid, expired, or doesn't have the right permissions
        return ErrorResponse(
            error="Invalid API key. Please check your key and try again.",
            error_type="invalid_key",
        )

    except RateLimitError:
        # The user has hit the provider's rate limit (common with free tiers)
        return ErrorResponse(
            error="Rate limit reached. Please wait a moment and try again.",
            error_type="rate_limit",
        )

    except APITimeoutError:
        # The LLM provider took too long to respond
        return ErrorResponse(
            error="The request timed out. The LLM provider might be experiencing high traffic. Please try again.",
            error_type="timeout",
        )

    except APIConnectionError:
        # Can't reach the provider's API (network issue, provider down, etc.)
        return ErrorResponse(
            error="Could not connect to the LLM provider. Please check your internet connection and try again.",
            error_type="connection_error",
        )

    except APIStatusError as e:
        # Catch-all for other API errors (500s, 503s, etc.)
        print(f"❌ API error: {e.status_code} — {e.message}")
        return ErrorResponse(
            error=f"The LLM provider returned an error (HTTP {e.status_code}). Please try again later.",
            error_type="api_error",
        )

    except ValueError as e:
        # Invalid provider name or config issue
        return ErrorResponse(
            error=str(e),
            error_type="invalid_provider",
        )

    except Exception as e:
        # Something completely unexpected happened
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return ErrorResponse(
            error="An unexpected error occurred. Please try again.",
            error_type="unknown",
        )

# ──────────────────────────────────────────────────────
# Quick test — run the full pipeline end-to-end
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    # Try loading from .env for local testing
    test_key = os.getenv("OPENROUTER_API_KEY")

    if not test_key:
        print("⚠️  No OPENROUTER_API_KEY found in .env")
        print("   Create a .env file with: OPENROUTER_API_KEY=sk-or-v1-your-key-here")
        print("   Or pass it directly below for a quick test.")
        sys.exit(1)

    test_queries = [
        "Approximately, How many people visited Epstein's island?",
        "What did Maxwell say in her deposition?",
        "Tell me about the flight logs for flight N909JE.",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"🔍 Query: {query}")
        print(f"{'='*60}")

        result = run_rag_query(
            question=query,
            api_key=test_key,
            provider="OpenRouter",
        )

        if isinstance(result, ErrorResponse):
            print(f"❌ Error ({result.error_type}): {result.error}")
        else:
            print(f"\n📝 Answer:\n{result.answer}")
            print(f"\n📄 Sources ({len(result.sources)}):")
            for s in result.sources:
                print(f"   - {s.source_file} (chunk {s.chunk_index}, score: {s.relevance_score})")
            print(f"\n⏱️  Retrieval: {result.retrieval_time_ms}ms | Generation: {result.generation_time_ms}ms")