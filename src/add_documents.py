"""Add new or updated articles to the existing vector store without a full rebuild.

Input is a JSON list of articles with the same fields as articles.json:
    article_number, ar_text, text_en (optional), citation (optional), book, chapter, ...

Usage:
    python -m src.add_documents --input data/new/new_articles.json

Running it twice with the same file gives the same index (it is idempotent):
old chunks of an article are removed before the new ones are written.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from src.chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS, make_chunks


def validate_articles(articles: list[dict]) -> None:
    if not isinstance(articles, list) or not articles:
        raise ValueError("Input must be a non-empty JSON list of articles.")
    seen: set[int] = set()
    for item in articles:
        number = item.get("article_number")
        if not isinstance(number, int):
            raise TypeError(f"article_number must be an integer, got {number!r}")
        if number in seen:
            raise ValueError(f"Duplicate article_number in input: {number}")
        seen.add(number)
        if not (item.get("ar_text") or "").strip():
            raise ValueError(f"Article {number} has empty ar_text.")


def add_articles(
    articles: list[dict],
    collection,
    embed: Callable[[list[str]], list[list[float]]],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> int:
    """Chunk, embed and upsert the articles. Returns the number of chunks written."""
    from src.indexing import build_embedding_text, build_metadata

    validate_articles(articles)
    chunks = make_chunks(articles, max_chars=max_chars, overlap_chars=overlap_chars)
    if not chunks:
        return 0

    # Remove the old chunks of these articles first, so a shorter new version
    # does not leave stale chunks behind.
    for number in sorted({c["article_number"] for c in chunks}):
        collection.delete(where={"article_number": number})

    documents = [build_embedding_text(chunk) for chunk in chunks]
    embeddings = embed(documents)

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=documents,
        embeddings=embeddings,
        metadatas=[build_metadata(chunk) for chunk in chunks],
    )
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP_CHARS)
    args = parser.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer

    from src.config import EMBEDDING_MODEL, VECTOR_COLLECTION_NAME, VECTOR_DB_DIR

    articles = json.loads(args.input.read_text(encoding="utf-8"))

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    collection = client.get_collection(VECTOR_COLLECTION_NAME)
    before = collection.count()

    model = SentenceTransformer(EMBEDDING_MODEL)
    written = add_articles(
        articles,
        collection,
        embed=lambda texts: model.encode(texts, normalize_embeddings=True).tolist(),
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )

    print(f"Chunks written: {written}")
    print(f"Collection size: {before} -> {collection.count()}")


if __name__ == "__main__":
    main()
