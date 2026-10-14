"""Validate the extracted corpus before chunking and embedding."""

import json
import re
from pathlib import Path

import pytest

ARTICLES_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "processed" / "articles.json"
)

# Articles the source PDF never prints (flagged as repealed, never deleted).
REPEALED_RANGES = [(54, 80), (389, 417)]

# Anything longer than this probably means a failed split.
MAX_TEXT_CHARS = 6000

# Standalone "ال" or "اال" means a lam-alef ligature was extracted reversed
# ("لا" -> "ال", "الا" -> "اال").
BROKEN_LAM_ALEF = re.compile(r"(?<![\u0621-\u064A])(?:ال|اال)(?![\u0621-\u064A])")

# Checked by hand: these articles contain the real verb "آل" (passed to),
# which normalization turns into a standalone "ال". Not an extraction bug.
LEGIT_STANDALONE_AL = {439, 900, 901, 947}


@pytest.fixture(scope="module")
def articles():
    assert ARTICLES_PATH.exists(), f"Missing file: {ARTICLES_PATH}"
    with ARTICLES_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    assert isinstance(data, list) and data, "articles.json must be a non-empty list"
    return data


def test_article_numbers_are_integers(articles):
    bad = [
        a.get("article_number")
        for a in articles
        if not isinstance(a.get("article_number"), int)
    ]
    assert not bad, f"Non-integer article numbers: {bad[:10]}"


def test_article_numbers_are_unique(articles):
    numbers = [a["article_number"] for a in articles]
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    assert not duplicates, f"Duplicate article numbers: {duplicates[:20]}"


def test_article_numbers_are_contiguous(articles):
    numbers = sorted(a["article_number"] for a in articles)
    expected = set(range(numbers[0], numbers[-1] + 1))
    missing = sorted(expected - set(numbers))
    assert not missing, f"Unexplained gaps in article numbers: {missing[:30]}"


def test_every_record_has_arabic_text(articles):
    bad = []
    for a in articles:
        if a.get("is_repealed") or a.get("ar_missing_in_source"):
            continue
        if not (isinstance(a.get("ar_text"), str) and a["ar_text"].strip()):
            bad.append(a["article_number"])
    assert not bad, f"Articles with empty ar_text: {bad[:20]}"


def test_missing_arabic_articles_have_english(articles):
    bad = [
        a["article_number"]
        for a in articles
        if a.get("ar_missing_in_source") and not (a.get("text_en") or "").strip()
    ]
    assert not bad, f"Flagged articles with no English text either: {bad}"


def test_every_live_article_has_english_text(articles):
    bad = [
        a["article_number"]
        for a in articles
        if not a.get("is_repealed") and not (a.get("text_en") or "").strip()
    ]
    assert not bad, f"Articles with empty text_en: {bad[:20]}"


def test_no_record_is_too_long(articles):
    too_long = [
        (a["article_number"], len(a.get("ar_text", "")))
        for a in articles
        if len(a.get("ar_text", "")) > MAX_TEXT_CHARS
    ]
    assert not too_long, f"Suspiciously long records (failed split?): {too_long[:10]}"


def test_repealed_articles_are_flagged(articles):
    by_number = {a["article_number"]: a for a in articles}
    unflagged = [
        n
        for start, end in REPEALED_RANGES
        for n in range(start, end + 1)
        if n in by_number and not by_number[n].get("is_repealed")
    ]
    assert not unflagged, f"Repealed articles not flagged: {unflagged}"
    assert any(a.get("is_repealed") for a in articles), "No repealed article is flagged"


def test_no_broken_lam_alef(articles):
    bad = [
        a["article_number"]
        for a in articles
        if a["article_number"] not in LEGIT_STANDALONE_AL
        and BROKEN_LAM_ALEF.search(a.get("ar_text", ""))
    ]
    assert not bad, f"Articles with reversed lam-alef: {bad[:20]} (total {len(bad)})"