Got it — here's the entire README as one single unbroken code block:

```markdown
# 🕵️ AskJeffrey

A RAG pipeline implementation for querying the Jeffrey Epstein Files using AI — built on the [Epstein Files 20K](https://huggingface.co/datasets/teyler/epstein-files-20k) dataset from Hugging Face.

> 🔗 **[Try the Live Demo →](#)** *(coming soon)*

---

## ⚡ Quick Demo

Process 2M+ document lines → Get accurate, source-cited answers in seconds

**What it does:**
- Semantically chunks documents based on meaning, not character count
- Searches using both vector similarity AND keyword matching (hybrid search)
- Re-ranks results with a cross-encoder for maximum precision
- Generates grounded answers with source citations
- Users bring their own free API key — no server costs

---

## 🎯 Key Features

✅ **No Hallucinations** - Answers grounded solely in source documents
✅ **Semantic Chunking** - Context-aware splits where meaning shifts
✅ **Hybrid Search** - Vector (ChromaDB) + Keyword (BM25) retrieval
✅ **Cross-Encoder Re-ranking** - Precision filtering of retrieved chunks
✅ **Source Citations** - Every answer cites its source documents
✅ **BYOK (Bring Your Own Key)** - Users provide their own free Groq API key
✅ **Fast Processing** - ~1 second end-to-end query response
✅ **Interactive Chat UI** - Streamlit web interface with conversation history

---

## 🏗️ How It Works

### Four Simple Stages

**Stage 1: Data Preparation** *(offline, run once)*

```

Raw Documents (2.5M lines)

↓

Clean & Reconstruct

↓

Semantic Chunking

↓

Vector Embeddings + BM25 Index

```

**Stage 2: Hybrid Retrieval**

```

User Question

↓

Vector Search (ChromaDB) + Keyword Search (BM25)

↓

Reciprocal Rank Fusion → Top 15 Chunks

```

**Stage 3: Re-ranking**

```

Top 15 Chunks + Question

↓

Cross-Encoder Scoring

↓

Top 6 Most Relevant Chunks

```

**Stage 4: Grounded Answer**

```

Context + Question

↓

LLaMA 3.3 70B (via Groq)

↓

Answer with Source Citations

```

### Why Hybrid Search + Re-ranking?

**Typical Approach:** Pure vector similarity
→ Misses exact names, dates, and keywords

**AskJeffrey's Approach:** Vector + BM25 + Cross-Encoder
→ Catches both semantic meaning AND exact matches, then precision-filters the results

---

## ✨ What Makes This Different?

| Feature | Typical RAG Projects | AskJeffrey |
|---|---|---|
| Chunking | Fixed character splits | **Semantic chunking** (meaning-based) |
| Search | Vector similarity only | **Hybrid** (vector + BM25 keyword) |
| Ranking | No re-ranking | **Cross-encoder re-ranking** |
| Embeddings | MiniLM (384d) | **BGE-base-en-v1.5** (768d) |
| API Key | Hardcoded / server-side | **BYOK** (user provides their own) |
| Citations | None | **Source documents cited** in answers |

---

## 📦 Installation

### Requirements
- Python 3.11+
- A free Groq API key ([get one here](https://console.groq.com))

### Setup (5 minutes)

**1. Clone repository**

```

git clone https://github.com/imdvz/AskJeffrey.git

cd AskJeffrey

```

**2. Create virtual environment**

```

python -m venv venv

source venv/bin/activate  # Windows: venvScriptsactivate

```

**3. Install dependencies**

```

pip install -r requirements.txt

```

---

## 🚀 Getting Started

### Run the Data Pipeline (first time only)

```

# Step 1: Download raw data

python ingest/download_[dataset.py](http://dataset.py)

# Step 2: Clean and reconstruct documents

python ingest/clean_[dataset.py](http://dataset.py)

# Step 3: Semantic chunking

python ingest/chunk_[dataset.py](http://dataset.py)

# Step 4: Generate embeddings + BM25 index

python ingest/embed_[chunks.py](http://chunks.py)

```

### Launch the App

```

streamlit run [app.py](http://app.py)

```

UI opens at: `http://localhost:8501`

**That's it!** Paste your Groq API key in the sidebar and start asking questions.

---

## 📚 Project Structure

```

AskJeffrey/

├── ingest/                        # Data processing pipeline

│   ├── download_[dataset.py](http://dataset.py)        # Download from Hugging Face

│   ├── clean_[dataset.py](http://dataset.py)           # Clean & reconstruct docs

│   ├── chunk_[dataset.py](http://dataset.py)           # Semantic chunking

│   └── embed_[chunks.py](http://chunks.py)            # Embed & build BM25 index

├── retrieval/                     # Retrieval logic

│   ├── hybrid_[retriever.py](http://retriever.py)        # Vector + BM25 hybrid search

│   └── [reranker.py](http://reranker.py)                # Cross-encoder re-ranking

├── core/                          # Core RAG chain

│   └── rag_[chain.py](http://chain.py)               # Orchestrates retrieval → LLM

├── api/                           # FastAPI backend (optional)

│   ├── [main.py](http://main.py)                    # API routes

│   ├── [models.py](http://models.py)                  # Pydantic models

│   └── [prompts.py](http://prompts.py)                 # Prompt templates

├── [app.py](http://app.py)                         # Streamlit frontend

├── [config.py](http://config.py)                      # Central configuration

├── requirements.txt               # Python dependencies

└── .env.example                   # Environment template

```

---

## 🔐 Bring Your Own Key (BYOK)

This app does **not** use a server-side API key. Every user provides their own free Groq API key:

- 🔒 Your key is **never stored** — it lives only in your browser session
- 🚫 Your key is **never logged** — it's sent directly to Groq's API and nowhere else
- 🗑️ When you close the tab, your key is **gone**

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** [Teyler/Epstein Files 20K](https://huggingface.co/datasets/teyler/epstein-files-20k) on Hugging Face
- **Embeddings:** [Sentence Transformers](https://www.sbert.net/)
- **Vector DB:** [ChromaDB](https://www.trychroma.com/)
- **Keyword Search:** [rank-bm25](https://github.com/dorianbrown/rank_bm25)
- **Re-ranker:** [Cross-Encoders](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **LLM Inference:** [Groq](https://groq.com/)
- **Framework:** [LangChain](https://langchain.com/)
- **UI:** [Streamlit](https://streamlit.io/)

---

## 📞 Support

**Get Help:**
- 📝 [Open an Issue](https://github.com/imdvz/AskJeffrey/issues)
- 💬 [Start a Discussion](https://github.com/imdvz/AskJeffrey/discussions)

---

## ⚠️ Disclaimer

This project is built for **research, transparency, and educational purposes**. All data is sourced from public records. Users are responsible for complying with applicable laws and ethical guidelines when using this system.

---
```
