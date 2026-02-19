import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────
# Paths
# ──────────────────────────────────────
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw.json")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned.json")
CHUNKS_DATA_PATH = os.path.join(DATA_DIR, "chunks.json")
BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
BM25_CHUNKS_PATH = os.path.join(DATA_DIR, "bm25_chunks.json")
CHROMA_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "epstein"

# ──────────────────────────────────────
# Dataset
# ──────────────────────────────────────
HF_DATASET_NAME = "teyler/epstein-files-20k"

# ──────────────────────────────────────
# Embedding Model
# ──────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# ──────────────────────────────────────
# Chunking
# ──────────────────────────────────────
MAX_CHUNK_SIZE = 1000
MIN_DOC_LENGTH = 100

# ──────────────────────────────────────
# Embedding Batch
# ──────────────────────────────────────
EMBED_BATCH_SIZE = 1000

# ──────────────────────────────────────
# Retrieval
# ──────────────────────────────────────
VECTOR_FETCH_K = 40
BM25_FETCH_K = 40
HYBRID_TOP_K = 15
RRF_K = 60

# ──────────────────────────────────────
# Re-ranking
# ──────────────────────────────────────
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 6

# ──────────────────────────────────────
# LLM Providers (BYOK)
# ──────────────────────────────────────
PROVIDERS = {
    "OpenRouter": {
        "model": "openai/gpt-oss-120b:free",
        "base_url": "https://openrouter.ai/api/v1",
    },
    # Easy to add more later:
    # "Groq": {
    #     "model": "llama-3.3-70b-versatile",
    #     "base_url": "https://api.groq.com/openai/v1",
    # },
}

DEFAULT_PROVIDER = "OpenRouter"

# ──────────────────────────────────────
# LLM Settings
# ──────────────────────────────────────
LLM_TEMPERATURE = 0
LLM_MAX_TOKENS = 500