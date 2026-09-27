from fastapi.testclient import TestClient

from legal_rag.api import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert "documents_indexed" in body


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_with_no_index_returns_safe_fallback():
    response = client.post("/ask", json={"question": "ما هي شروط صحة العقد؟"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
