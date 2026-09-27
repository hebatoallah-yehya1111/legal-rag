"""FastAPI application exposing the legal RAG system.

Run locally with:
    uvicorn legal_rag.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from legal_rag import config
from legal_rag.rag import LegalRAG
from legal_rag.schema import AskRequest, AskResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag = LegalRAG()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the corpus and build the index once at startup.

    TODO: currently tolerant of a missing corpus file so the service can
    boot before Step 0 (ingestion) is finished; documents_indexed will be 0.
    """
    try:
        rag.load_corpus(config.PROCESSED_JSON_PATH)
        rag.build_index()
    except FileNotFoundError:
        logger.warning(
            "No processed corpus found at %s yet - run ingestion first.",
            config.PROCESSED_JSON_PATH,
        )
    yield


app = FastAPI(
    title="Arabic Legal Document Q&A",
    description="RAG system over the Egyptian Civil Code with cited answers.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", documents_indexed=rag.documents_indexed)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    answer, sources = rag.ask(request.question)
    return AskResponse(answer=answer, sources=sources)
