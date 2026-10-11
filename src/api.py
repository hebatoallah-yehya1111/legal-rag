"""FastAPI application for the Egyptian Civil Code RAG system."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        rag = get_rag()
        rag.top_k = request.top_k
        return rag.answer(request.question)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Check the application logs.",
        ) from exc
