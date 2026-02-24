
# AskJeffrey

A **Retrieval-Augmented Generation (RAG)** pipeline for querying the **Jeffrey Epstein Files** using AI — built on the **Epstein Files 20K** dataset from Hugging Face.

- Dataset: https://huggingface.co/datasets/teyler/epstein-files-20k

> 🔗 **Live Demo:** *(coming soon)*

---

## ⚡ Quick Demo

**Process 2M+ document lines → Get accurate, source-cited answers in seconds**

### What it does
- Semantic chunking (splits by meaning, not character count)
- Hybrid retrieval: **vector similarity + keyword matching**
- Cross-encoder **re-ranking** for precision
- Grounded answers with **source citations**
- **BYOK (Bring Your Own Key):** users provide their own free OpenAI API key

---

## 🎯 Key Features

- ✅ **Grounded Answers (No Hallucinations)** — responses are derived only from retrieved source text
- ✅ **Semantic Chunking** — context-aware splits where meaning shifts
- ✅ **Hybrid Search** — ChromaDB (vector) + BM25 (keyword)
- ✅ **Cross-Encoder Re-ranking** — filters results for maximum relevance
- ✅ **Source Citations** — every answer includes citations to the underlying chunks
- ✅ **BYOK (Bring Your Own Key)** — no server-side API key required
- ✅ **Fast Response** — ~1 second end-to-end query time (typical)
- ✅ **Interactive Chat UI** — Streamlit interface with conversation history

---

## 🏗️ How It Works

### Four Stages (Simple Pipeline)

#### Stage 1 — Data Preparation *(offline, run once)*

```text
Raw Documents (2.5M lines)
        ↓
Clean & Reconstruct
        ↓
Semantic Chunking
        ↓
Vector Embeddings + BM25 Index
```

#### Stage 2 — Hybrid Retrieval

```text
User Question
        ↓
Vector Search (ChromaDB) + Keyword Search (BM25)
        ↓
Reciprocal Rank Fusion → Top 15 Chunks
```

#### Stage 3 — Re-ranking

```text
Top 15 Chunks + Question
        ↓
Cross-Encoder Scoring
        ↓
Top 6 Most Relevant Chunks
```

#### Stage 4 — Grounded Answer

```text
Context + Question
        ↓
gpt-oss-120b (via OpenRouter)
        ↓
Answer with Source Citations
```

---

## 🧠 Why Hybrid Search + Re-ranking?

**Typical RAG:** vector similarity only
→ often misses **exact names, dates, identifiers**, and keyword-heavy queries.

**AskJeffrey:** vector + BM25 + cross-encoder
→ captures **semantic meaning + exact matches**, then **precision-filters** results before generation.

---

## ✨ What Makes This Different?

| Feature    | Typical RAG Projects  | AskJeffrey                            |
| ---------- | --------------------- | ------------------------------------- |
| Chunking   | Fixed-size splits     | **Semantic chunking** (meaning-based) |
| Search     | Vector only           | **Hybrid** (vector + BM25 keyword)    |
| Ranking    | No re-ranking         | **Cross-encoder re-ranking**          |
| Embeddings | MiniLM (384d)         | **BGE-base-en-v1.5** (768d)           |
| API Key    | Hardcoded/server-side | **BYOK** (user provides their own)    |
| Citations  | Often missing         | **Always included**                   |

---

## 📦 Installation

### Requirements

* Python **3.11+**
* A free **OpenAI API key**: [https://openrouter.ai/](https://openrouter.ai/)

### Setup (5 minutes)

#### 1) Clone the repository

```bash
git clone https://github.com/imdvz/AskJeffrey.git
cd AskJeffrey
```

#### 2) Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

### Run the Data Pipeline *(first time only)*

```bash
# Step 1: Download raw data
python ingest/download_dataset.py

# Step 2: Clean and reconstruct documents
python ingest/clean_dataset.py

# Step 3: Semantic chunking
python ingest/chunk_dataset.py

# Step 4: Generate embeddings + BM25 index
python ingest/embed_chunks.py
```

### Launch the App

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

Paste your **OpenAI API key** in the sidebar and start asking questions.

---

## 📚 Project Structure

```text
AskJeffrey/
├── ingest/                         # Data processing pipeline
│   ├── download_dataset.py          # Download from Hugging Face
│   ├── clean_dataset.py             # Clean & reconstruct docs
│   ├── chunk_dataset.py             # Semantic chunking
│   └── embed_chunks.py              # Embed & build BM25 index
│
├── retrieval/                       # Retrieval logic
│   ├── hybrid_retriever.py          # Vector + BM25 hybrid search
│   └── reranker.py                  # Cross-encoder re-ranking
│
├── core/                            # Core RAG chain
│   └── rag_chain.py                 # Orchestrates retrieval → LLM
│
├── api/                             # FastAPI backend (optional)
│   ├── main.py                      # API routes
│   ├── models.py                    # Pydantic models
│   └── prompts.py                   # Prompt templates
│
├── app.py                           # Streamlit frontend
├── config.py                        # Central configuration
├── requirements.txt                 # Python dependencies
└── .env.example                     # Environment template
```

---

## 🔐 Bring Your Own Key (BYOK)

This app does **not** use a server-side API key. Every user provides their own free OpenAI API key:

* 🔒 Key is **not stored** (browser session only)
* 🚫 Key is **not logged**
* 🗑️ Closing the tab clears it

---

## 📜 License

Licensed under the **MIT License** — see `LICENSE`.

---

## 🙏 Acknowledgments

* **Dataset:** Teyler / Epstein Files 20K (Hugging Face)
  [https://huggingface.co/datasets/teyler/epstein-files-20k](https://huggingface.co/datasets/teyler/epstein-files-20k)
* **Embeddings:** Sentence Transformers — [https://www.sbert.net/](https://www.sbert.net/)
* **Vector DB:** ChromaDB — [https://www.trychroma.com/](https://www.trychroma.com/)
* **Keyword Search:** rank-bm25 — [https://github.com/dorianbrown/rank_bm25](https://github.com/dorianbrown/rank_bm25)
* **Re-ranking:** Cross-Encoders — [https://www.sbert.net/docs/cross_encoder/usage/usage.html](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
* **LLM Inference:** OpenAI (gpt-oss-120b) — [https://openrouter.ai/](https://openrouter.ai/)
* **Framework:** LangChain — [https://langchain.com/](https://langchain.com/)
* **UI:** Streamlit — [https://streamlit.io/](https://streamlit.io/)

---

## 📞 Support

* 📝 Issues: [https://github.com/imdvz/AskJeffrey/issues](https://github.com/imdvz/AskJeffrey/issues)
* 💬 Discussions: [https://github.com/imdvz/AskJeffrey/discussions](https://github.com/imdvz/AskJeffrey/discussions)

---

## ⚠️ Disclaimer

Built for **research, transparency, and educational purposes**.
All data is sourced from public records. Users are responsible for complying with applicable laws and ethical guidelines.
