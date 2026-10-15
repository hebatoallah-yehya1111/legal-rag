"""Unit tests for LegalRAG. Groq and the retriever are replaced by fakes."""

import pytest

from src import rag as rag_module


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.last_top_k = None

    def search(self, question, top_k):
        self.last_top_k = top_k
        return self.results


class FakeGroq:
    """Records calls so tests can check that the model was (not) used."""

    calls = 0

    def __init__(self, api_key=None):
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        FakeGroq.calls += 1

        class _Message:
            content = "إجابة تجريبية من المادة 147."

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        return _Response()


ARTICLE_147 = {
    "article_number": 147,
    "citation": "Egyptian Civil Code, Article 147",
    "document": "العقد شريعة المتعاقدين",
    "distance": 0.21,
    "is_repealed": False,
}


def build_rag(monkeypatch, results):
    retriever = FakeRetriever(results)
    FakeGroq.calls = 0
    monkeypatch.setattr(rag_module, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(rag_module, "Groq", FakeGroq)
    monkeypatch.setattr(rag_module, "Retriever", lambda: retriever)
    return rag_module.LegalRAG(top_k=5), retriever


def test_no_retrieved_articles_returns_safe_fallback(monkeypatch):
    rag, _ = build_rag(monkeypatch, results=[])

    result = rag.answer("سؤال لا علاقة له بالقانون")

    assert result["answer"]
    assert result["citations"] == []
    assert result["retrieved_articles"] == []
    assert FakeGroq.calls == 0  # the model is never asked to guess


def test_answer_returns_citations_and_contexts(monkeypatch):
    rag, _ = build_rag(monkeypatch, results=[ARTICLE_147])

    result = rag.answer("ما هي شروط العقد؟")

    assert result["answer"]
    assert result["citations"] == ["Egyptian Civil Code, Article 147"]
    assert result["retrieved_contexts"] == ["العقد شريعة المتعاقدين"]
    assert FakeGroq.calls == 1


def test_top_k_override_is_used_for_that_call_only(monkeypatch):
    rag, retriever = build_rag(monkeypatch, results=[ARTICLE_147])

    rag.answer("ما هي شروط العقد؟", top_k=3)
    assert retriever.last_top_k == 3

    rag.answer("ما هي شروط العقد؟")
    assert retriever.last_top_k == 5  # back to the default, nothing leaked


def test_missing_api_key_fails_early(monkeypatch):
    monkeypatch.setattr(rag_module, "GROQ_API_KEY", "")
    with pytest.raises(ValueError):
        rag_module.LegalRAG()