import pytest

from legal_rag.ingestion import is_repealed, normalize_arabic_digits, validate_articles
from legal_rag.schema import Article


def test_normalize_arabic_digits():
    assert normalize_arabic_digits("١٤٧") == "147"


def test_is_repealed_flags_known_range():
    assert is_repealed(60) is True
    assert is_repealed(147) is False


def make_article(number: int, text: str = "نص المادة") -> Article:
    return Article(
        article_number=number,
        text_ar=text,
        citation=f"Egyptian Civil Code, Article {number}",
    )


def test_validate_articles_passes_on_contiguous_numbers():
    articles = [make_article(n) for n in range(1, 6)]
    validate_articles(articles)  # should not raise


def test_validate_articles_fails_on_gap():
    articles = [make_article(1), make_article(2), make_article(5)]
    with pytest.raises(AssertionError):
        validate_articles(articles)


def test_validate_articles_fails_on_empty_text():
    articles = [make_article(1, text="")]
    with pytest.raises(AssertionError):
        validate_articles(articles)
