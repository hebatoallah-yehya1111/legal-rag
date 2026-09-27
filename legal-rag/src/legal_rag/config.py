"""Centralized configuration. Values can be overridden with environment variables."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_PDF_PATH = Path(os.getenv("RAW_PDF_PATH", PROJECT_ROOT / "data/raw/egyptian_civil_code.pdf"))
PROCESSED_JSON_PATH = Path(
    os.getenv("PROCESSED_JSON_PATH", PROJECT_ROOT / "data/processed/articles.json")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
VECTOR_DB_DIR = Path(os.getenv("VECTOR_DB_DIR", PROJECT_ROOT / "data/processed/vector_store"))
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "egyptian_civil_code")

CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "by_article")

LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "placeholder-llm")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "legal-rag-chunking")

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

RAGAS_FAITHFULNESS_THRESHOLD = float(os.getenv("RAGAS_FAITHFULNESS_THRESHOLD", "0.75"))
