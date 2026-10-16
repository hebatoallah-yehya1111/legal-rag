"""Tests for the Hit@k / MRR maths. No embedding model or index is needed."""

import pytest

from src.evaluate_retrieval import evaluate, first_rank, summarize, unique_in_order


def test_first_rank_is_one_based():
    assert first_rank([5, 9, 12], {9}) == 2


def test_first_rank_is_none_when_missing():
    assert first_rank([5, 9, 12], {99}) is None


def test_unique_in_order_removes_duplicate_chunks():
    assert unique_in_order([7, 7, 3, None, 7, 4]) == [7, 3, 4]


def test_summarize_hit_at_k_and_mrr():
    # ranks: found first, found 4th, never found
    summary = summarize([1, 4, None])
    assert summary["hit_at_1"] == pytest.approx(1 / 3)
    assert summary["hit_at_3"] == pytest.approx(1 / 3)
    assert summary["hit_at_5"] == pytest.approx(2 / 3)
    assert summary["mrr"] == pytest.approx((1 + 0.25) / 3)


def test_evaluate_skips_out_of_scope_questions():
    gold = [
        {"id": "q1", "question": "a", "language": "ar", "scope": "in_scope", "gold_articles": [10]},
        {"id": "q2", "question": "b", "language": "en", "scope": "out_of_scope", "gold_articles": []},
    ]

    def fake_search(question, k):
        return [{"article_number": 3}, {"article_number": 10}]

    results = evaluate(gold, fake_search)

    assert [r["id"] for r in results] == ["q1"]
    assert results[0]["rank"] == 2