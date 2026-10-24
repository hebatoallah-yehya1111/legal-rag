"""BentoML wrapper around the RAG pipeline, with an async /ask and a streaming endpoint.

Run locally:
    pip install bentoml
    bentoml serve src.bento_service:LegalRAGService --port 3000

Then:
    curl -X POST http://localhost:3000/ask -H "Content-Type: application/json" \
         -d '{"question": "ما القانون الذي يحكم الالتزامات غير التعاقدية؟", "top_k": 5}'

Tokens appear progressively from /ask_stream.

Note: the generative model is a hosted API (Groq), so there is no local vLLM
server in this project (vLLM needs a GPU). The LLM call is isolated in
src/rag.py, so a vLLM server with an OpenAI-compatible endpoint could replace it later.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import bentoml

from src.rag import LegalRAG


@bentoml.service(resources={"cpu": "2"}, traffic={"timeout": 120})
class LegalRAGService:
    def __init__(self) -> None:
        self.rag = LegalRAG()

    @bentoml.api
    async def ask(self, question: str, top_k: int = 5) -> dict:
        if not question or len(question.strip()) < 3:
            raise ValueError("question must have at least 3 characters")
        # The pipeline is blocking (embedding + HTTP call), so run it in a thread.
        result = await asyncio.to_thread(self.rag.answer, question.strip(), top_k)
        sources = list(dict.fromkeys(result.get("citations", [])))
        return {"answer": result["answer"], "sources": sources}

    @bentoml.api
    async def ask_stream(self, question: str, top_k: int = 5) -> AsyncGenerator[str, None]:
        if not question or len(question.strip()) < 3:
            raise ValueError("question must have at least 3 characters")
        iterator = iter(self.rag.stream_answer(question.strip(), top_k))
        sentinel = object()
        while True:
            token = await asyncio.to_thread(next, iterator, sentinel)
            if token is sentinel:
                break
            yield token
