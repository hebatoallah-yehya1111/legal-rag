"""RAG pipeline: index articles into a vector store, and answer questions.

Chunking strategy: by article (not fixed token windows) - see ingestion.py
and the project handbook for why. Each chunk keeps its article_number so
answers can cite "Egyptian Civil Code, Article N" instead of a chunk id.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from legal_rag import config
from legal_rag.schema import Article

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    article: Article
    score: float


class LegalRAG:
    """Wraps embedding, vector store retrieval, and answer generation.

    TODO:
      - swap the placeholder embedder for sentence-transformers
        (config.EMBEDDING_MODEL)
      - swap the placeholder vector store for Chroma at config.VECTOR_DB_DIR
      - swap the placeholder generator for a vLLM-served model
    """

    def __init__(self) -> None:
        self._articles: list[Article] = []
        self._index_ready = False

    def load_corpus(self, json_path: Path = config.PROCESSED_JSON_PATH) -> list[Article]:
        with Path(json_path).open(encoding="utf-8") as f:
            raw = json.load(f)
        self._articles = [Article(**record) for record in raw]
        return self._articles

    def build_index(self) -> None:
        """Embed each article and upsert into the vector store.

        TODO: replace with real embedding + Chroma upsert.
        """
        if not self._articles:
            raise RuntimeError("Call load_corpus() before build_index().")
        logger.info("Indexing %d articles (placeholder - no real embedding yet).", len(self._articles))
        self._index_ready = True

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top_k most relevant articles for a question.

        TODO: replace with a real vector similarity search. Placeholder
        currently returns an empty list so the API can be wired and tested
        end-to-end before the retrieval logic exists.
        """
        if not self._index_ready:
            logger.warning("Index not built yet - returning no results.")
            return []
        return []

    def generate_answer(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Call the generative LLM with the retrieved context.

        TODO: replace with a real vLLM call. Must refuse to answer (rather
        than guess) when chunks is empty, since an ungrounded answer here
        is exactly the hallucination risk this project exists to prevent.
        """
        if not chunks:
            return "لا تتوفر معلومات كافية في القانون المدني للإجابة على هذا السؤال."
        raise NotImplementedError("Generation not yet implemented.")

    def ask(self, question: str, top_k: int = 5) -> tuple[str, list[str]]:
        chunks = self.retrieve(question, top_k=top_k)
        answer = self.generate_answer(question, chunks)
        sources = [c.article.display_citation() for c in chunks]
        return answer, sources

    @property
    def documents_indexed(self) -> int:
        return len(self._articles)
