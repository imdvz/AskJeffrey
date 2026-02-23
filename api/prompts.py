"""
Prompt templates for the RAG chain.

This file defines how we talk to the LLM. Prompt engineering is arguably the
most impactful part of a RAG system — a great retrieval pipeline means nothing
if the LLM doesn't know how to USE the retrieved chunks properly.

We define:
1. SYSTEM_PROMPT    → Sets the LLM's persona, rules, and behavior
2. CONTEXT_TEMPLATE → Formats retrieved chunks into a structured context block
3. USER_TEMPLATE    → Wraps the user's question with the context

Design principles:
- Be explicit about what the LLM should and shouldn't do
- Force citation of sources — reduces hallucination significantly
- Handle "no relevant context" gracefully — better to say "I don't know" than hallucinate
- Keep formatting instructions clear — the LLM's output goes directly to the user
"""

# ──────────────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are **AskJeffrey**, an AI research assistant specializing in the publicly released \
Jeffrey Epstein court documents and related files.

## Your Role
- You help users explore, understand, and find information within the Epstein files.
- You answer questions ONLY based on the provided context (retrieved document chunks).
- You are factual, precise, and careful with sensitive information.

## Rules You Must Follow

### 1. Stick to the Context
- Base your answers STRICTLY on the provided context chunks.
- If the context doesn't contain enough information to fully answer the question, \
say so clearly. Do NOT guess or fill in gaps with outside knowledge.
- If the context is completely irrelevant to the question, respond with: \
"I couldn't find relevant information about this in the available documents."

### 2. Cite Your Sources
- When referencing information from the context, cite the source file in brackets.
- Format: [SOURCE_FILE_NAME]
- Example: "According to the flight logs [TEXT-001-HOUSE_OVERSIGHT_010757.txt], ..."
- If multiple chunks support a claim, cite all of them.

### 3. Handle Sensitive Content Responsibly
- These are real legal documents involving serious allegations.
- Present information factually without editorializing or sensationalizing.
- Use phrases like "according to the documents" or "the records indicate" \
rather than making definitive claims yourself.
- Distinguish between allegations, testimony, and established facts.

### 4. Formatting
- Use markdown formatting for readability.
- Use bullet points for lists of names, dates, or events.
- Use bold for key names, dates, and locations.
- Keep answers concise but thorough — aim for 3-8 sentences for simple questions, \
longer for complex ones.
- If a question has multiple parts, address each part separately.

### 5. When You're Not Confident
- If the retrieved chunks have low relevance to the question, acknowledge this.
- Say something like: "The available documents contain limited information about this, \
but here's what I found: ..."
- NEVER make up information. NEVER hallucinate facts about real people.\
"""

# ──────────────────────────────────────────────────────
# Context Template
# ──────────────────────────────────────────────────────

CONTEXT_TEMPLATE = """\
## Retrieved Document Chunks

The following chunks were retrieved from the Epstein files and ranked by relevance. \
Use them to answer the user's question. Cite the source file when referencing information.

{chunks}\
"""

# Each individual chunk gets formatted like this before being joined together
CHUNK_TEMPLATE = """\
---
**[Chunk {index}]** Source: `{source_file}` | Relevance: {relevance}
{text}
---\
"""

# ──────────────────────────────────────────────────────
# User Message Template
# ──────────────────────────────────────────────────────

USER_TEMPLATE = """\
{context}

## User's Question
{question}\
"""

# ──────────────────────────────────────────────────────
# No Context Template
# ──────────────────────────────────────────────────────
# Used when retrieval returns nothing or all chunks have very low confidence

NO_CONTEXT_RESPONSE = """\
I couldn't find relevant information about this in the available Epstein files. \
This could mean:

- The documents don't contain information about this specific topic.
- The question might need to be rephrased with different keywords.
- The relevant information might be in a part of the files that wasn't included in this dataset.

Try rephrasing your question, or ask about a specific person, event, date, or document ID.\
"""

# ──────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────

def format_chunks_as_context(chunks: list[dict]) -> str:
    """
    Format a list of reranked chunks into the context block for the LLM.

    Takes the output from Reranker.rerank() and converts it into a clean,
    structured text block that the LLM can easily parse and cite.

    Each chunk includes:
    - An index number (for the LLM to reference internally)
    - The source filename (for citation)
    - A relevance indicator (high/medium/low based on rerank score)
    - The actual text

    Args:
        chunks: List of chunk dicts from Reranker.rerank(), each containing
                'text', 'metadata', and 'rerank_score' keys.

    Returns:
        Formatted context string ready to be inserted into USER_TEMPLATE.
    """
    if not chunks:
        return ""

    formatted_chunks = []

    for i, chunk in enumerate(chunks):
        # Convert rerank score to a human-readable relevance label
        # This helps the LLM understand which chunks to trust more
        score = chunk.get('rerank_score', 0)
        if score >= 5.0:
            relevance = "🟢 High"
        elif score >= 2.0:
            relevance = "🟡 Medium"
        elif score >= 0.0:
            relevance = "🟠 Low"
        else:
            relevance = "🔴 Very Low"

        formatted = CHUNK_TEMPLATE.format(
            index=i + 1,
            source_file=chunk['metadata'].get('source_file', 'Unknown'),
            relevance=relevance,
            text=chunk['text'],
        )
        formatted_chunks.append(formatted)

    return CONTEXT_TEMPLATE.format(chunks="\n".join(formatted_chunks))

def build_user_message(question: str, chunks: list[dict]) -> str:
    """
    Build the complete user message that gets sent to the LLM.

    Combines the formatted context (retrieved chunks) with the user's question
    into a single message string.

    Args:
        question: The user's original question.
        chunks:   List of reranked chunk dicts.

    Returns:
        Complete user message string for the LLM.
    """
    context = format_chunks_as_context(chunks)
    return USER_TEMPLATE.format(context=context, question=question)

def has_sufficient_context(chunks: list[dict], threshold: float = 0.0) -> bool:
    """
    Check if we have enough relevant context to attempt an answer.

    If ALL chunks have negative rerank scores, the retrieval pipeline is
    essentially saying "nothing in the database really matches this query."
    In that case, it's better to return a graceful "I don't know" rather
    than forcing the LLM to hallucinate from irrelevant context.

    Args:
        chunks:    List of reranked chunk dicts.
        threshold: Minimum rerank score for the best chunk. Default 0.0.

    Returns:
        True if at least the top chunk meets the confidence threshold.
    """
    if not chunks:
        return False

    best_score = max(chunk.get('rerank_score', 0) for chunk in chunks)
    return best_score >= threshold

# ──────────────────────────────────────────────────────
# Quick test — preview what the LLM will actually see
# ──────────────────────────────────────────────────────

if __name__ == '__main__':
    # Simulate some reranked chunks to see the formatted output
    mock_chunks = [
        {
            'text': 'Epstein flew to the island on March 5. He was accompanied by two associates.',
            'metadata': {'source_file': 'TEXT-001-HOUSE_OVERSIGHT_010486.txt', 'doc_id': 'abc123', 'chunk_index': 0},
            'rerank_score': 6.3,
        },
        {
            'text': 'Maxwell testified that she had no knowledge of any illegal activities on the island.',
            'metadata': {'source_file': 'IMAGES-002-HOUSE_OVERSIGHT_013341.txt', 'doc_id': 'def456', 'chunk_index': 2},
            'rerank_score': 4.1,
        },
        {
            'text': 'The flight manifest showed departures from Teterboro Airport.',
            'metadata': {'source_file': 'TEXT-001-HOUSE_OVERSIGHT_010757.txt', 'doc_id': 'ghi789', 'chunk_index': 5},
            'rerank_score': -1.2,
        },
    ]

    print("=" * 60)
    print("📝 SYSTEM PROMPT")
    print("=" * 60)
    print(SYSTEM_PROMPT)

    print("\n" + "=" * 60)
    print("💬 USER MESSAGE (what the LLM sees)")
    print("=" * 60)
    user_msg = build_user_message("Who visited Epstein's island?", mock_chunks)
    print(user_msg)

    print("\n" + "=" * 60)
    print("🔍 CONTEXT SUFFICIENT?")
    print("=" * 60)
    print(f"With all chunks: {has_sufficient_context(mock_chunks)}")
    print(f"With only low-confidence chunk: {has_sufficient_context([mock_chunks[2]])}")