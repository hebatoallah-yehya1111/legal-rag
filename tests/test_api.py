"""API tests. A fake RAG object is used, so no Groq key, model or index is needed."""

import pytest
from fastapi.testclient import TestClient

from src import api


class FakeRAG:
    def __init__(self):
        self.last_top_k = None

    def answer(self, question, top_k=5):
        self.last_top_k = top_k
        return {
            "question": question,
            "answer": "العقد شريعة المتعاقدين.",
            "citations": [
                "Egyptian Civil Code, Article 147",
                "Egyptian Civil Code, Article 147",
                "Egyptian Civil Code, Article 148",
            ],
            "retrieved_articles": [
                {"article_number": 147, "citation": "Egyptian Civil Code, Article 147"}
            ],
        }

    def stream_answer(self, question, top_k=5):
        self.last_top_k = top_k
        yield "إجابة "
        yield "متدفقة"


class BrokenRAG:
    def answer(self, question, top_k=5):
        raise RuntimeError("secret internal failure")


@pytest.fixture
def fake_rag():
    return FakeRAG()


@pytest.fixture
def client(fake_rag):
    api.app.dependency_overrides[api.get_rag] = lambda: fake_rag
    api.app.dependency_overrides[api.get_documents_indexed] = lambda: 1148
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health_reports_indexed_documents(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "documents_indexed": 1148}


def test_health_is_unhealthy_when_index_is_empty(client):
    api.app.dependency_overrides[api.get_documents_indexed] = lambda: 0
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unhealthy"


def test_ask_returns_answer_and_unique_article_sources(client):
    response = client.post("/ask", json={"question": "ما هي شروط العقد؟"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"]
    assert body["sources"] == [
        "Egyptian Civil Code, Article 147",
        "Egyptian Civil Code, Article 148",
    ]


def test_ask_passes_top_k_to_the_rag(client, fake_rag):
    client.post("/ask", json={"question": "ما هي شروط العقد؟", "top_k": 3})
    assert fake_rag.last_top_k == 3


def test_ask_rejects_empty_question(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_whitespace_only_question(client):
    response = client.post("/ask", json={"question": "     "})
    assert response.status_code == 422


def test_ask_rejects_missing_question(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_rejects_top_k_out_of_range(client):
    response = client.post("/ask", json={"question": "ما هي شروط العقد؟", "top_k": 50})
    assert response.status_code == 422


def test_ask_hides_internal_errors(client):
    api.app.dependency_overrides[api.get_rag] = lambda: BrokenRAG()
    response = client.post("/ask", json={"question": "ما هي شروط العقد؟"})
    assert response.status_code == 500
    assert "secret internal failure" not in response.text

def test_stream_endpoint_emits_server_sent_events(client, fake_rag):
    response = client.post("/ask/stream", json={"question": "ما هي شروط العقد؟"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"token": "إجابة "' in response.text
    assert '"token": "متدفقة"' in response.text
    assert "event: done" in response.text


def test_stream_endpoint_passes_top_k(client, fake_rag):
    client.post("/ask/stream", json={"question": "ما هي شروط العقد؟", "top_k": 3})
    assert fake_rag.last_top_k == 3


def test_provider_rate_limit_returns_429():
    class LimitedRAG:
        def answer(self, question, top_k=5):
            raise api.RateLimitError("too many tokens")

    api.app.dependency_overrides[api.get_rag] = lambda: LimitedRAG()
    try:
        response = TestClient(api.app).post("/ask", json={"question": "ما هو العقد؟"})
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 429
    assert "Retry-After" in response.headers


def test_get_rag_builds_one_instance_under_concurrency(monkeypatch):
    import threading
    import time

    created = []

    class SlowRAG:
        def __init__(self):
            created.append(1)
            time.sleep(0.2)

    monkeypatch.setattr(api, "LegalRAG", SlowRAG)
    monkeypatch.setattr(api, "_rag_instance", None)

    threads = [threading.Thread(target=api.get_rag) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 1