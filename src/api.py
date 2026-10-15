"""FastAPI application for the Egyptian Civil Code RAG system."""

import logging
from functools import lru_cache
from typing import Any

import chromadb
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import VECTOR_COLLECTION_NAME, VECTOR_DB_DIR
from src.rag import LegalRAG

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Egyptian Civil Code RAG API",
    description="Arabic and English legal question answering.",
    version="0.1.0",
)


class AskRequest(BaseModel):
    # Strip spaces first, so "    " is rejected like an empty question.
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_articles: list[dict[str, Any]] = Field(default_factory=list)


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
        logger.exception("Could not read the vector store")
        return 0


@app.get("/health")
def health(documents_indexed: int = Depends(get_documents_indexed)):
    # An empty or missing index is reported as unhealthy (503), so the
    # Docker healthcheck catches a container that has no vector store.
    if documents_indexed == 0:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "documents_indexed": 0},
        )
    return {"status": "healthy", "documents_indexed": documents_indexed}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, rag: LegalRAG = Depends(get_rag)):
    try:
        # top_k is passed per request. Never set it on the shared rag object,
        # because concurrent requests would overwrite each other's value.
        result = rag.answer(request.question, top_k=request.top_k)
    except Exception as exc:
        logger.exception("Failed to answer the question")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Check the application logs.",
        ) from exc

    raw_sources = result.get("sources") or result.get("citations") or []
    # Several chunks can come from one article, so remove duplicates.
    sources = list(dict.fromkeys(raw_sources))

    return AskResponse(
        answer=result["answer"],
        sources=sources,
        retrieved_articles=result.get("retrieved_articles", []),
    )