"""FastAPI application for the Egyptian Civil Code RAG system."""

import json
import logging
import threading
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import VECTOR_COLLECTION_NAME, VECTOR_DB_DIR
from src.rag import LegalRAG

logger = logging.getLogger(__name__)

try:
    from groq import RateLimitError
except ImportError:  # groq is not installed in some test environments
    class RateLimitError(Exception):
        """Placeholder so the except clause below works without groq."""

try:
    from langfuse import observe
except ImportError:  # observability is optional for local development/tests
    def observe(*, name: str, **_kwargs):
        def decorator(func):
            return func
        return decorator

def _trace(**fields: Any) -> None:
    """Attach small explicit fields to the current Langfuse span.

    @observe must not capture the function arguments automatically: they include
    the LegalRAG object, which holds the embedding model, and serializing it
    copies the model weights and crashes the process. Tracing must never break
    a request, so every error here is swallowed.
    """
    try:
        from langfuse import get_client

        get_client().update_current_span(**fields)
    except Exception:
        logger.debug("Langfuse span update skipped", exc_info=True)


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


_rag_lock = threading.Lock()
_rag_instance: LegalRAG | None = None


def get_rag() -> LegalRAG:
    """Return the shared LegalRAG, creating it once.

    The lock matters: several requests can arrive before the first one has
    finished loading the embedding model. Without it each request loads the
    model at the same time, and torch fails with "Cannot copy out of meta tensor".
    """
    global _rag_instance
    if _rag_instance is None:
        with _rag_lock:
            if _rag_instance is None:
                _rag_instance = LegalRAG()
    return _rag_instance


def get_documents_indexed() -> int:
    """Return the actual indexed chunk count, or zero if unavailable."""
    try:
        import chromadb

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
@observe(name="legal-rag.ask", capture_input=False, capture_output=False)
def ask(request: AskRequest, rag: LegalRAG = Depends(get_rag)):
    _trace(input={"question": request.question, "top_k": request.top_k})
    try:
        # top_k is passed per request. Never set it on the shared rag object,
        # because concurrent requests would overwrite each other's value.
        result = rag.answer(request.question, top_k=request.top_k)
    except RateLimitError as exc:
        # The hosted LLM refused the call because of its token or request limit.
        # Report it as 429 so clients and load tests can tell it from a real failure.
        logger.warning("LLM provider rate limit reached")
        headers = getattr(getattr(exc, "response", None), "headers", None) or {}
        retry_after = str(headers.get("retry-after", "20"))
        raise HTTPException(
            status_code=429,
            detail="The language model is rate limited. Try again shortly.",
            headers={"Retry-After": retry_after},
        ) from exc
    except Exception as exc:
        logger.exception("Failed to answer the question")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question. Check the application logs.",
        ) from exc

    raw_sources = result.get("sources") or result.get("citations") or []
    # Several chunks can come from one article, so remove duplicates.
    sources = list(dict.fromkeys(raw_sources))
    _trace(output={"answer": result["answer"], "sources": sources})

    return AskResponse(
        answer=result["answer"],
        sources=sources,
        retrieved_articles=result.get("retrieved_articles", []),
    )

@app.post("/ask/stream")
@observe(name="legal-rag.ask-stream", capture_input=False, capture_output=False)
def ask_stream(request: AskRequest, rag: LegalRAG = Depends(get_rag)):
    """Stream answer tokens as Server-Sent Events (SSE)."""
    _trace(input={"question": request.question, "top_k": request.top_k})
    def events() -> Iterator[str]:
        try:
            for token in rag.stream_answer(request.question, top_k=request.top_k):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception:
            logger.exception("Failed to stream an answer")
            yield (
                "event: error\ndata: "
                + json.dumps(
                    {"detail": "Unable to process the question. Check application logs."},
                    ensure_ascii=False,
                )
                + "\n\n"
            )

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})