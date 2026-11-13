"""Settings shared by more than one module. Change a value here, not in the scripts."""
import os
from pathlib import Path

from opensearchpy import OpenSearch

# ---- Paths ----
# Anchored to this file, so scripts work no matter which folder you run them from.
#   ROOT       -> /Users/allen/.../finsearch
#   RAW_DIR    -> finsearch/data/raw      (downloaded HTML)
#   TEXT_DIR   -> finsearch/data/text     (plain text, extracted sections)
#   CHUNKS_DIR -> finsearch/data/chunks   (passage records)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TEXT_DIR = DATA_DIR / "text"
CHUNKS_DIR = DATA_DIR / "chunks"

# ---- OpenSearch ----
# Env vars let the API container reach OpenSearch by its compose service name ("opensearch")
# instead of localhost; running scripts on your Mac needs no env vars.
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
INDEX_NAME = "filings"


def get_client():
    # Used by indexing/create_index.py, indexing/index_chunks.py, search/retrieval.py
    return OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}])


# ---- Models (see "Design decisions" in README before changing) ----
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"               # models/embeddings.py
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"    # models/cross_encoder.py
LLM_MODEL = "qwen3:8b"                                   # rag/answer.py; qwen3:4b if 8B is too slow
