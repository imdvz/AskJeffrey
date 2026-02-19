# AskJeffrey

```markdown
# 🕵️ AskJeffrey

**Ask AI-powered questions about the Jeffrey Epstein Files.**

A RAG (Retrieval-Augmented Generation) pipeline built on the [Epstein Files 20K](https://huggingface.co/datasets/teyler/epstein-files-20k) dataset from Hugging Face — featuring semantic chunking, hybrid search, cross-encoder re-ranking, and a Streamlit chat interface.

> 🔗 **[Try the Live Demo →](#)** *(coming soon)*

---

## ✨ What Makes This Different?

| Feature | Typical RAG Projects | AskJeffrey |
|---|---|---|
| Chunking | Fixed character splits | **Semantic chunking** (splits where meaning shifts) |
| Search | Vector similarity only | **Hybrid search** (vector + BM25 keyword) |
| Ranking | No re-ranking | **Cross-encoder re-ranking** for precision |
| Embeddings | MiniLM (384d) | **BGE-base-en-v1.5** (768d) |
| API Key | Hardcoded / server-side | **BYOK** (Bring Your Own Key) — user provides their own |
| Citations | None | **Source documents cited** in every answer |

---

## 🏗️ How It Works

```

User asks a question (Streamlit UI)

↓

Hybrid Retrieval (ChromaDB + BM25 → Reciprocal Rank Fusion)

↓

Cross-Encoder Re-ranking (top 6 most relevant chunks)

↓

LLM generates grounded answer (LLaMA 3.3 via Groq)

↓

Answer + source citations returned to user

```

---

## 📦 Tech Stack

- **LLM**: LLaMA 3.3 70B via [Groq](https://groq.com)
- **Framework**: [LangChain](https://langchain.com)
- **Vector DB**: [ChromaDB](https://www.trychroma.com)
- **Embeddings**: [BGE-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) (Sentence Transformers)
- **Keyword Search**: BM25 (rank-bm25)
- **Re-ranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Frontend**: [Streamlit](https://streamlit.io)
- **Backend**: [FastAPI](https://fastapi.tiangolo.com) (optional)
- **Dataset**: [Epstein Files 20K](https://huggingface.co/datasets/teyler/epstein-files-20k)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- A free Groq API key → [Get one here](https://console.groq.com)

### Setup

```

# Clone the repo

git clone https://github.com/imdvz/AskJeffrey.git

cd AskJeffrey

# Create virtual environment

python -m venv venv

source venv/bin/activate  # Windows: venvScriptsactivate

# Install dependencies

pip install -r requirements.txt

```

### Run the Data Pipeline (first time only)

```

python ingest/download_[dataset.py](http://dataset.py)    # Download dataset

python ingest/clean_[dataset.py](http://dataset.py)       # Clean & reconstruct documents

python ingest/chunk_[dataset.py](http://dataset.py)       # Semantic chunking

python ingest/embed_[chunks.py](http://chunks.py)        # Generate embeddings + BM25 index

```

### Launch the App

```

streamlit run [app.py](http://app.py)

```

Open `http://localhost:8501`, paste your Groq API key in the sidebar, and start asking questions!

---

## 📁 Project Structure

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

- Your key is **never stored** — it lives only in your browser session
- Your key is **never logged** — it's sent directly to Groq's API and nowhere else
- When you close the tab, your key is gone

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This project is built for **research, transparency, and educational purposes**. All data is sourced from public records. Users are responsible for complying with applicable laws and ethical guidelines when using this system.

---

## 🙏 Acknowledgments

- Dataset: [Teyler/Epstein Files 20K](https://huggingface.co/datasets/teyler/epstein-files-20k) on Hugging Face
- Inspired by [AnkitNayak-eth/EpsteinFiles-RAG](https://github.com/AnkitNayak-eth/EpsteinFiles-RAG)
```
