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
MAX_CHUNK_SIZE = 1100
MIN_DOC_LENGTH = 120

# ──────────────────────────────────────
# Embedding Batch
# ──────────────────────────────────────
EMBED_BATCH_SIZE = 64

# ──────────────────────────────────────
# Retrieval
# ──────────────────────────────────────
VECTOR_FETCH_K = 80 # Increased from 40 to 80 to provide more candidates for RRF re-ranking
BM25_FETCH_K = 80 # Increased from 40 to 80 to provide more candidates for RRF re-ranking
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

# ──────────────────────────────────────
# Query Intent + Hybrid Weighting
# ──────────────────────────────────────

# Enable/disable island query expansion for BM25
ENABLE_ISLAND_QUERY_EXPANSION = True

# Terms we append to BM25 queries ONLY when "island intent" is implied
ISLAND_EXPANSION_TERMS = [
    "island",
    "st james",
    "st. james",
    "little st james",
    "great st james",
    "virgin islands",
    "u.s. virgin islands",
    "usvi",
    "st thomas",
    "caribbean",
]

# If any of these appear, we consider "island intent" implied
ISLAND_INTENT_KEYWORDS = [
    "island",
    "st james",
    "st. james",
    "little st james",
    "great st james",
    "virgin islands",
    "u.s. virgin islands",
    "usvi",
    "st thomas",
    "caribbean",
]

# If "epstein" appears AND any of these verbs appear, we consider island intent implied
ISLAND_INTENT_VERBS = [
    "visit",
    "visited",
    "visiting",
    "guest",
    "guests",
    "went",
    "travel",
    "traveled",
    "fly",
    "flew",
    "flight",
    "stayed",
    "stay",
    "staying",
    "trip",
    "villa",
    "compound",
]

# Enable/disable identifier query boost (BM25 dominates)
ENABLE_IDENTIFIER_BOOST = True

# Identifier patterns (tail numbers, exhibit IDs, doc IDs, etc.)
IDENTIFIER_PATTERNS = [
    r"\b[A-Z]\d{3}[A-Z]{2}\b",      # e.g., N908JE-like variants (user-provided pattern)
    r"\bN\d{3,6}[A-Z]{0,3}\b",      # e.g., N908JE, N2123, N123AB
    r"\b[A-Z]{1,4}-\d{2,8}\b",      # e.g., JE-1045, EX-12, DOC-2020
    r"\b\d{2,4}-\d{2,6}\b",         # e.g., 0124-0420
]

# Default RRF weights
DENSE_WEIGHT_DEFAULT = 1.0
SPARSE_WEIGHT_DEFAULT = 1.0

# Weights for identifier-like queries (BM25 should dominate)
DENSE_WEIGHT_IDENTIFIER = 0.5
SPARSE_WEIGHT_IDENTIFIER = 2.0