
"""FastAPI application for the Egyptian Civil Code RAG system."""

from functools import lru_cache

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import VECTOR_DB_DIR, VECTOR_COLLECTION_NAME
from src.rag import LegalRAG

app = FastAPI(
    title="Egyptian Civil Code RAG API",
    description="Arabic and English legal question answering.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


@lru_cache(maxsize=1)
def get_rag() -> LegalRAG:
    return LegalRAG()


def get_documents_indexed() -> int:
    """Return the actual indexed chunk count, or zero if unavailable."""
    try:
        client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        collection = client.get_collection(name=VECTOR_COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents_indexed": get_documents_indexed(),
    }


@app.post("/ask")
def ask(request: AskRequest):
    try:
        rag = get_rag()
        rag.top_k = request.top_k
        result = rag.answer(request.question)

        # Preserve existing response fields while providing the API's
        # expected sources field.
        if "sources" not in result:
            result["sources"] = result.get("citations", [])

        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Check the application logs.",
        ) from exc
