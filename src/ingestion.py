"""Convert the Egyptian Civil Code PDF into structured, per-article JSON."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pdfplumber

from schema import Article

logger = logging.getLogger(__name__)

ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
WESTERN_DIGITS = "0123456789"
_DIGIT_TRANSLATION = str.maketrans(ARABIC_INDIC_DIGITS, WESTERN_DIGITS)

KNOWN_REPEALED_RANGES: list[tuple[int, int]] = [
    (54, 80),
    (389, 417),
]

KNOWN_MISSING_ARABIC_MARKERS: set[int] = {1022}


def is_expected_gap(article_number: int) -> bool:
    return is_repealed(article_number) or article_number in KNOWN_MISSING_ARABIC_MARKERS

COLUMN_THRESHOLD = 300

ARABIC_ARTICLE_LINE_RE = re.compile(r"^مادة[\s()]*([٠-٩]+)[\s()]*$")
ENGLISH_ARTICLE_LINE_RE = re.compile(r"^Article\s+(\d+)\s*$")

BOOK_KEYWORD = "الكتاب"
CHAPTER_KEYWORD = "الباب"
SECTION_KEYWORD = "الفصل"
HEADING_KEYWORDS = (BOOK_KEYWORD, CHAPTER_KEYWORD, SECTION_KEYWORD)

SOURCE_PAGE_OFFSET = 18

ENGLISH_BOOK = "Obligations Generally"
ENGLISH_CHAPTER = "Sources of Obligations"
ENGLISH_SECTION = "Contracts"
ENGLISH_TOPIC_LINE_RE = re.compile(r"^(\d+)\.\s+(.+?)\s*:?$")


def normalize_english_heading(text: str) -> str:
    text = text.strip().title()
    for word in ("Of", "Or", "And", "The"):
        if word in {"Of", "Or", "And"}:
            text = re.sub(rf"\b{word}\b", word.lower(), text)
    return text


def is_arabic_digits_only(text: str) -> bool:
    return all(c in ARABIC_INDIC_DIGITS for c in text)


def fix_arabic_word(text: str) -> str:
    if is_arabic_digits_only(text):
        return text
    segments = re.findall(r"[٠-٩]+|[^٠-٩]+", text)
    fixed_segments = [
        seg if is_arabic_digits_only(seg) else seg[::-1]
        for seg in reversed(segments)
    ]
    return "".join(fixed_segments)


def group_words_into_lines(words: list[dict]) -> dict[int, list[dict]]:
    lines: dict[int, list[dict]] = {}
    for w in words:
        key = round(w["top"])
        lines.setdefault(key, []).append(w)
    return lines


def build_arabic_line_text(line_words: list[dict]) -> str:
    sorted_words = sorted(line_words, key=lambda w: w["x0"], reverse=True)
    fixed = [fix_arabic_word(w["text"]) for w in sorted_words]
    return " ".join(fixed)


def build_english_line_text(line_words: list[dict]) -> str:
    sorted_words = sorted(line_words, key=lambda w: w["x0"])
    return " ".join(w["text"] for w in sorted_words)


def normalize_arabic_digits(text: str) -> str:
    return text.translate(_DIGIT_TRANSLATION)


def normalize_arabic_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"[\u064B-\u0652]", "", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_repealed(article_number: int) -> bool:
    return any(start <= article_number <= end for start, end in KNOWN_REPEALED_RANGES)


def extract_raw_blocks(pdf_path: Path) -> list[dict]:
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_number, page in enumerate(pdf.pages):
            words = page.extract_words()
            english_words = [w for w in words if w["x0"] < COLUMN_THRESHOLD]
            arabic_words = [w for w in words if w["x0"] >= COLUMN_THRESHOLD]

            arabic_lines = group_words_into_lines(arabic_words)
            english_lines = group_words_into_lines(english_words)

            arabic_text = "\n".join(
                build_arabic_line_text(arabic_lines[top])
                for top in sorted(arabic_lines.keys())
            )
            english_text = "\n".join(
                build_english_line_text(english_lines[top])
                for top in sorted(english_lines.keys())
            )

            blocks.append(
                {
                    "page_number": page_number,
                    "text_ar": arabic_text,
                    "text_en": english_text,
                }
            )

            if page_number % 20 == 0:
                logger.info("Extracted page %d/%d", page_number, total)

    return blocks


def _collect_english_metadata_by_article(
    blocks: list[dict],
) -> tuple[dict[int, str], dict[int, dict[str, str | None]]]:
    texts_by_number: dict[int, list[str]] = {}
    metadata_by_number: dict[int, dict[str, str | None]] = {}

    current_number: int | None = None
    current_book: str | None = None
    current_chapter: str | None = None
    current_section: str | None = None
    current_topic: str | None = None
    pending_heading: str | None = None

    lines: list[str] = []
    for block in blocks:
        lines.extend(
            line.strip()
            for line in block["text_en"].split("\n")
            if line.strip()
        )

    i = 0
    while i < len(lines):
        line = lines[i]

        if re.fullmatch(r"BOOK\s+[IVXLCDM]+", line, re.IGNORECASE):
            pending_heading = "book"
            i += 1
            continue
        if re.fullmatch(r"CHAPTER\s+[IVXLCDM]+", line, re.IGNORECASE):
            pending_heading = "chapter"
            i += 1
            continue
        if re.fullmatch(r"Section\s+[IVXLCDM]+", line, re.IGNORECASE):
            pending_heading = "section"
            i += 1
            continue

        if pending_heading:
            heading = normalize_english_heading(line)
            if pending_heading == "book":
                current_book = heading
            elif pending_heading == "chapter":
                current_chapter = heading
            else:
                current_section = heading
            pending_heading = None
            i += 1
            continue

        match = ENGLISH_ARTICLE_LINE_RE.match(line)
        if match:
            current_number = int(match.group(1))
            texts_by_number.setdefault(current_number, [])
            metadata_by_number[current_number] = {
                "book": current_book,
                "chapter": current_chapter,
                "section": current_section,
                "topic": current_topic,
            }
            i += 1
            continue

        topic_match = ENGLISH_TOPIC_LINE_RE.match(line)
        if topic_match:
            # A legal article can contain numbered clauses such as
            # "6. Any group of persons ...". Treat a numbered line as a
            # topic ONLY when the next non-empty extracted line is an
            # Article marker. This matches the PDF structure where a real
            # topic heading is immediately followed by its Article number.
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if ENGLISH_ARTICLE_LINE_RE.match(next_line):
                current_topic = topic_match.group(2).strip()
                i += 1
                continue

        if current_number is not None:
            texts_by_number.setdefault(current_number, []).append(line)

        i += 1

    english_by_number = {
        num: " ".join(lines).strip()
        for num, lines in texts_by_number.items()
    }
    return english_by_number, metadata_by_number


def _insert_repealed_placeholders(articles: list[Article]) -> list[Article]:
    """Add a placeholder entry for every article number inside
    KNOWN_REPEALED_RANGES that has no real entry (the source PDF never
    prints these articles at all - see KNOWN_REPEALED_RANGES above).

    Each placeholder borrows book/chapter/section from the nearest real
    article (the preceding one if there is one, otherwise the following
    one) so retrieval filters by heading still work for it, and carries a
    plain-language note instead of invented legal text.
    """
    by_number = {a.article_number: a for a in articles}
    sorted_numbers = sorted(by_number)

    for start, end in KNOWN_REPEALED_RANGES:
        for number in range(start, end + 1):
            if number in by_number:
                continue

            preceding = max(
                (n for n in sorted_numbers if n < number), default=None
            )
            following = min(
                (n for n in sorted_numbers if n > number), default=None
            )
            context = by_number.get(preceding) or by_number.get(following)

            placeholder = Article(
                article_number=number,
                book=context.book if context else None,
                chapter=context.chapter if context else None,
                section=context.section if context else None,
                topic=context.topic if context else None,
                ar_text=(
                    f"المادة {number} ملغاة ولا تظهر في نص القانون المصدر."
                ),
                text_en=f"Article {number} has been repealed and does not appear in the source text.",
                is_repealed=True,
                source_page=context.source_page if context else None,
                citation=f"Egyptian Civil Code, Article {number} (repealed)",
            )
            by_number[number] = placeholder
            sorted_numbers = sorted(by_number)

    return [by_number[n] for n in sorted(by_number)]


def parse_articles(raw_blocks: list[dict]) -> list[Article]:
    english_by_number, metadata_by_number = _collect_english_metadata_by_article(
        raw_blocks
    )

    articles: list[Article] = []
    current_number: int | None = None
    current_page: int | None = None
    current_body_lines: list[str] = []

    def flush_current_article() -> None:
        if current_number is None:
            return

        meta = metadata_by_number.get(current_number, {})
        text_ar = normalize_arabic_text(" ".join(current_body_lines))

        articles.append(
            Article(
                article_number=current_number,
                book=meta.get("book"),
                chapter=meta.get("chapter"),
                section=meta.get("section"),
                topic=meta.get("topic"),
                ar_text=text_ar,
                text_en=english_by_number.get(current_number),
                is_repealed=is_repealed(current_number),
                source_page=(
                    current_page + SOURCE_PAGE_OFFSET
                    if current_page is not None
                    else None
                ),
                citation=f"Egyptian Civil Code, Article {current_number}",
            )
        )

    for block in raw_blocks:
        for line in block["text_ar"].split("\n"):
            line = line.strip()
            if not line:
                continue

            match = ARABIC_ARTICLE_LINE_RE.match(line)
            if match:
                flush_current_article()
                current_number = int(normalize_arabic_digits(match.group(1)))
                current_page = block["page_number"] + 1
                current_body_lines = []
                continue

            if current_number is not None:
                current_body_lines.append(line)

    flush_current_article()
    return _insert_repealed_placeholders(articles)


def validate_articles(articles: list[Article]) -> None:
    numbers = sorted(a.article_number for a in articles)
    all_gaps = [n for n in range(numbers[0], numbers[-1] + 1) if n not in numbers]
    unexplained_gaps = [n for n in all_gaps if not is_expected_gap(n)]
    assert not unexplained_gaps, f"Unexplained gaps in article numbers: {unexplained_gaps}"

    for article in articles:
        assert article.ar_text.strip(), f"Article {article.article_number} has empty ar_text"
        assert len(article.ar_text) < 5000, (
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