"""Create retrieval chunks from the structured Egyptian Civil Code corpus."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 4000
DEFAULT_OVERLAP_CHARS = 300


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = []

    for paragraph in text.split("\n"):
        paragraph = re.sub(r"[ \t]+", " ", paragraph).strip()
        if paragraph:
            paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def split_long_text(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split long article text while preferring paragraph boundaries."""
    text = normalize_text(text)

    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            overlap = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            # A single paragraph is larger than max_chars.
            start = 0

            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                chunks.append(paragraph[start:end].strip())

                if end >= len(paragraph):
                    break

                start = max(0, end - overlap_chars)

            current = ""

    if current:
        chunks.append(current)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def make_chunks(
    articles: list[dict],
    max_chars: int,
    overlap_chars: int,
) -> list[dict]:
    chunks: list[dict] = []

    for article in articles:
        article_number = article["article_number"]
        ar_text = normalize_text(article.get("ar_text") or "")
        text_en = normalize_text(article.get("text_en") or "")

        if not ar_text:
            logger.warning(
                "Article %s has empty Arabic text; skipping.",
                article_number,
            )
            continue

        article_parts = split_long_text(
            ar_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        total_chunks = len(article_parts)

        for chunk_index, chunk_ar in enumerate(article_parts):
            chunk = {
                "chunk_id": f"article-{article_number}-chunk-{chunk_index}",
                "article_number": article_number,
                "chunk_index": chunk_index,
                "chunk_count": total_chunks,
                "book": article.get("book"),
                "chapter": article.get("chapter"),
                "section": article.get("section"),
                "topic": article.get("topic"),
                "ar_text": chunk_ar,
                "text_en": text_en,
                "is_repealed": article.get("is_repealed", False),
                "source_page": article.get("source_page"),
                "citation": article.get(
                    "citation",
                    f"Egyptian Civil Code, Article {article_number}",
                ),
            }

            chunks.append(chunk)

    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk the structured Egyptian Civil Code corpus."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to articles.json",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Path to chunks.json",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Maximum Arabic characters per chunk (default: {DEFAULT_MAX_CHARS})",
    )

    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
        help=f"Overlap between chunks (default: {DEFAULT_OVERLAP_CHARS})",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    args = parse_args()

    if args.max_chars <= 0:
        raise ValueError("--max-chars must be greater than 0")

    if args.overlap_chars < 0:
        raise ValueError("--overlap-chars cannot be negative")

    if args.overlap_chars >= args.max_chars:
        raise ValueError("--overlap-chars must be smaller than --max-chars")

    input_path = Path(args.input)
    output_path = Path(args.out)

    with input_path.open("r", encoding="utf-8") as file:
        articles = json.load(file)

    if not isinstance(articles, list):
        raise ValueError("Input JSON must contain a list of articles.")

    chunks = make_chunks(
        articles,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    multi_chunk_articles = sum(
        1
        for article in articles
        if len(
            split_long_text(
                normalize_text(article.get("ar_text") or ""),
                args.max_chars,
                args.overlap_chars,
            )
        )
        > 1
    )

    logger.info("Input articles: %d", len(articles))
    logger.info("Output chunks: %d", len(chunks))
    logger.info("Articles split into multiple chunks: %d", multi_chunk_articles)
    logger.info("max_chars=%d", args.max_chars)
    logger.info("overlap_chars=%d", args.overlap_chars)


if __name__ == "__main__":
    main()