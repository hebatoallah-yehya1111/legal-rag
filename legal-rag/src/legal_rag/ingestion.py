"""Convert the Egyptian Civil Code PDF into structured, per-article JSON.

This is Step 0 of the project: the PDF is raw input, not the corpus.
Run as a script:

    python -m legal_rag.ingestion --pdf data/raw/egyptian_civil_code.pdf \
        --out data/processed/articles.json

Known extraction hazards this module must handle (see project handbook):
  1. Bilingual two-column layout -> naive text dump interleaves AR/EN.
  2. RTL Arabic glyphs coming back reversed/disconnected -> needs reshaping.
  3. Article numbers in Arabic-Indic numerals (e.g. ١٤٧) -> normalize to int.
  4. Repealed articles (e.g. 54-80) -> flag, never silently drop.
  5. Book/Chapter/Section headings -> capture as metadata, not just body text.
  6. Diacritics / hamza / alef variants -> normalize consistently.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from legal_rag.schema import Article

logger = logging.getLogger(__name__)

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
_DIGIT_TRANSLATION = str.maketrans(ARABIC_INDIC_DIGITS, WESTERN_DIGITS)

# Filled in once the real PDF is available and the repealed ranges are confirmed
# against the source (e.g. Presidential Decree repealing Articles 54-80).
KNOWN_REPEALED_RANGES: list[tuple[int, int]] = [
    (54, 80),
]


def normalize_arabic_digits(text: str) -> str:
    """Convert Arabic-Indic digits (١٢٣) to Western digits (123)."""
    return text.translate(_DIGIT_TRANSLATION)


def normalize_arabic_text(text: str) -> str:
    """Normalize diacritics, hamza and alef variants for consistent embedding.

    TODO: extend once real text samples are available - current pass covers
    the most common alef/hamza/diacritic variants only.
    """
    if not text:
        return text
    text = re.sub(r"[\u064B-\u0652]", "", text)  # strip tashkeel (diacritics)
    text = re.sub(r"[إأآا]", "ا", text)  # normalize alef forms
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_repealed(article_number: int) -> bool:
    return any(start <= article_number <= end for start, end in KNOWN_REPEALED_RANGES)


def extract_raw_blocks(pdf_path: Path) -> list[dict]:
    """Extract raw per-page, column-aware text blocks from the PDF.

    TODO: implement with pdfplumber, using each word's bounding box (x0) to
    separate the Arabic (right) column from the English (left) column rather
    than reading in raw stream order, which interleaves the two languages.
    This is the step to validate manually against ~20 sample pages before
    trusting the full document.
    """
    raise NotImplementedError(
        "Column-aware extraction not yet implemented - needs the real PDF "
        "to validate against. See handbook Step 0.2."
    )


def parse_articles(raw_blocks: list[dict]) -> list[Article]:
    """Turn extracted blocks into Article records, one per legal article.

    TODO: implement heading detection for Book/Chapter/Section/Topic, and
    article-boundary detection (e.g. a line matching r'^مادة\\s+(\\d+)').
    """
    raise NotImplementedError("Article parsing not yet implemented.")


def validate_articles(articles: list[Article]) -> None:
    """Sanity checks before embedding. Raises AssertionError on failure."""
    numbers = sorted(a.article_number for a in articles)
    gaps = [n for n in range(numbers[0], numbers[-1] + 1) if n not in numbers]
    assert not gaps, f"Unexplained gaps in article numbers: {gaps}"

    for article in articles:
        assert article.text_ar.strip(), f"Article {article.article_number} has empty text_ar"
        assert len(article.text_ar) < 5000, (
            f"Article {article.article_number} text_ar looks too long "
            "(possible failed split)"
        )

    flagged = sum(1 for a in articles if a.is_repealed)
    logger.info("Validated %d articles (%d flagged as repealed).", len(articles), flagged)


def build_corpus(pdf_path: Path, out_path: Path) -> list[Article]:
    raw_blocks = extract_raw_blocks(pdf_path)
    articles = parse_articles(raw_blocks)
    validate_articles(articles)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([a.model_dump() for a in articles], f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d articles to %s", len(articles), out_path)
    return articles


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_corpus(args.pdf, args.out)


if __name__ == "__main__":
    main()
