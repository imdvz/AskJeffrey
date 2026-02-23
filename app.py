"""
AskJeffrey — Streamlit Frontend

A RAG-powered research assistant for exploring the publicly released
Jeffrey Epstein court documents. Users bring their own API key (BYOK)
to query the documents through a chat interface.

Structure:
  16a — pysqlite3 fix + page config
  16b — Sidebar (API key, provider, settings)
  16c — Welcome screen (no API key yet)
  16d — Chat interface (message history)
  16e — Handle new questions (RAG pipeline)
  16f — Error display
"""

# ──────────────────────────────────────────────────────
# 16a: pysqlite3 Fix + Page Config
# ──────────────────────────────────────────────────────
# MUST be the very first lines — before ANY other import.
# Streamlit Cloud ships with old SQLite 3.31, but ChromaDB
# needs 3.35+. This swaps in a newer version. Harmless on local dev.

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

# set_page_config MUST be the first st.* call — Streamlit enforces this
st.set_page_config(
    page_title="AskJeffrey",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Now safe to import heavy modules ─────────────────
# These imports trigger loading of the RAG pipeline (~500MB of models).
# On Streamlit Cloud, this happens once per server boot, not per user.
import time
from core.rag_chain import run_rag_query
from api.models import QueryResponse, ErrorResponse
from config import PROVIDERS, DEFAULT_PROVIDER, RERANK_TOP_N

# ──────────────────────────────────────────────────────
# 16b: Sidebar — API Key + Settings
# ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    # ─── API Key Input ────────────────────────────────
    # type="password" masks the key with dots — users feel safe pasting it.
    # The key lives ONLY in st.session_state — never saved to disk,
    # never sent anywhere except to the LLM provider's API.
    api_key = st.text_input(
        "🔑 OpenRouter API Key",
        type="password",
        placeholder="sk-or-v1-...",
        help="Your key is never stored. It's used only for LLM requests and discarded after.",
    )

    # ─── Provider Selector ────────────────────────────
    # For now there's only OpenRouter, but this dropdown makes it
    # trivial to add more providers later (Groq, Together, etc.)
    provider_options = list(PROVIDERS.keys())
    provider = st.selectbox(
        "🤖 LLM Provider",
        options=provider_options,
        index=provider_options.index(DEFAULT_PROVIDER),
    )

    # Show which model is being used — transparency for the user
    model_name = PROVIDERS[provider]["model"]
    st.caption(f"Model: `{model_name}`")

    # ─── Chunks Slider ────────────────────────────────
    # Lets power users control the context window:
    # Lower = faster + more focused answers
    # Higher = more context but potentially noisier
    top_k = st.slider(
        "📄 Source chunks to use",
        min_value=2,
        max_value=12,
        value=RERANK_TOP_N,
        help="How many document chunks to feed the LLM. More = broader context, fewer = more focused.",
    )

    st.divider()

    # ─── Info Box ─────────────────────────────────────
    st.info(
        "**Free to use!** Get your API key at "
        "[openrouter.ai/keys](https://openrouter.ai/keys)\n\n"
        "No credit card required. Make sure your "
        "[data policy](https://openrouter.ai/settings/privacy) "
        "allows free model usage."
    )

    # ─── Clear Chat Button ────────────────────────────
    # Resets the conversation so users can start fresh
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ──────────────────────────────────────────────────────
# 16c: Welcome Screen (No API Key Yet)
# ──────────────────────────────────────────────────────
# Gate the entire app behind the API key. If no key is provided,
# show a friendly onboarding screen and stop rendering everything else.

if not api_key:
    # App title
    st.title("🔍 AskJeffrey")
    st.markdown("##### Explore the Epstein Files with AI")

    st.divider()

    st.markdown(
        """
        **AskJeffrey** is a RAG-powered research assistant that lets you search and
        ask questions about the publicly released Jeffrey Epstein court documents.

        ### How it works
        1. **Hybrid search** finds the most relevant document chunks using both
           semantic understanding and keyword matching
        2. **Cross-encoder reranking** picks the best chunks that actually answer
           your question
        3. **An LLM** reads those chunks and generates a clear, cited answer

        ### Get started
        1. Get a **free** API key from [openrouter.ai/keys](https://openrouter.ai/keys)
        2. Paste it in the sidebar on the left
        3. Start asking questions!

        ---
        *Built with LangChain, ChromaDB, and Streamlit.
        Data sourced from publicly released court documents.*
        """
    )

    # Stop here — nothing below this renders until the user provides a key
    st.stop()

# ──────────────────────────────────────────────────────
# 16d: Chat Interface — Message History
# ──────────────────────────────────────────────────────

# App title (shown when API key is provided and chat is active)
st.title("🔍 AskJeffrey")

# Initialize message history in session state if it doesn't exist yet.
# session_state persists across reruns (every interaction triggers a rerun in Streamlit)
# but gets cleared when the user closes the tab or the server restarts.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render all previous messages from history.
# This loop runs on every rerun, rebuilding the chat UI from session state.
# Each message has a "role" (user/assistant) and "content" (the text).
# Assistant messages can also have "sources" and "timing" for the expander.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # If this is an assistant message with sources, show them in an expander
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📄 Sources ({len(message['sources'])})"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(
                        f"**[{i+1}]** `{source['source_file']}` "
                        f"(chunk {source['chunk_index']}, "
                        f"relevance: {source['relevance_score']:.2f})"
                    )
                    st.caption(source["text"][:300] + ("..." if len(source["text"]) > 300 else ""))
                    if i < len(message["sources"]) - 1:
                        st.divider()

        # Show timing info if available
        if message.get("timing"):
            st.caption(message["timing"])

# ──────────────────────────────────────────────────────
# 16e: Handle New Questions
# ──────────────────────────────────────────────────────

# Chat input box — always visible at the bottom of the page
question = st.chat_input("Ask a question about the Epstein files...")

if question:
    # ─── Display the user's message immediately ──────
    with st.chat_message("user"):
        st.markdown(question)

    # Save to history
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    # ─── Process the question ─────────────────────────
    with st.chat_message("assistant"):
        # Show a spinner while the pipeline runs
        # Users see this for ~5-10 seconds (retrieval + LLM generation)
        with st.spinner("Searching documents and generating answer..."):
            result = run_rag_query(
                question=question,
                api_key=api_key,
                provider=provider,
                top_k=top_k,
            )

        # ──────────────────────────────────────────────
        # 16f: Error Display
        # ──────────────────────────────────────────────
        # Check if the result is an error or a successful response.
        # We display errors differently based on type so users know
        # exactly what went wrong and how to fix it.

        if isinstance(result, ErrorResponse):
            # Map error types to appropriate Streamlit display methods
            if result.error_type == "invalid_key":
                st.error(f"🔑 {result.error}")
            elif result.error_type == "rate_limit":
                st.warning(f"⏳ {result.error}")
            elif result.error_type == "timeout":
                st.warning(f"⏱️ {result.error}")
            elif result.error_type == "connection_error":
                st.error(f"🌐 {result.error}")
            else:
                st.error(f"❌ {result.error}")

            # Save error to history so it persists across reruns
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ {result.error}",
            })

        else:
            # ─── Success! Display the answer ──────────
            st.markdown(result.answer)

            # ─── Sources Expander ─────────────────────
            # Shows the actual document chunks the answer was based on.
            # Users can verify claims by reading the original text.
            if result.sources:
                with st.expander(f"📄 Sources ({len(result.sources)})"):
                    for i, source in enumerate(result.sources):
                        st.markdown(
                            f"**[{i+1}]** `{source.source_file}` "
                            f"(chunk {source.chunk_index}, "
                            f"relevance: {source.relevance_score:.2f})"
                        )
                        st.caption(source.text[:300] + ("..." if len(source.text) > 300 else ""))
                        if i < len(result.sources) - 1:
                            st.divider()

            # ─── Timing Info ──────────────────────────
            # Small caption showing how fast each stage was.
            # Helps users (and you) understand performance.
            timing = (
                f"⏱️ Retrieval: {result.retrieval_time_ms:.0f}ms "
                f"| Generation: {result.generation_time_ms:.0f}ms "
                f"| Total: {result.retrieval_time_ms + result.generation_time_ms:.0f}ms"
            )
            st.caption(timing)

            # ─── Save to History ──────────────────────
            # We store the full response data so sources and timing
            # persist when Streamlit reruns (which happens on every interaction).
            # We convert SourceChunk Pydantic models → dicts for JSON serialization.
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "sources": [
                    {
                        "text": s.text,
                        "source_file": s.source_file,
                        "chunk_index": s.chunk_index,
                        "relevance_score": s.relevance_score,
                    }
                    for s in result.sources
                ],
                "timing": timing,
            })