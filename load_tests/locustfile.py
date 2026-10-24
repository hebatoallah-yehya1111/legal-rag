"""Load-test profile for the Legal RAG API.

Run the 50-user test from the handbook (one line, works in PowerShell):

    python -m locust -f load_tests/locustfile.py --host http://127.0.0.1:8000 --headless -u 50 -r 5 -t 3m --html reports/locust_report.html

The LLM is a hosted API (Groq). Its free tier has rate limits, so some /ask
calls can return an error. Connection failures, timeouts, HTTP 429 and other
HTTP errors are reported under separate messages, so throttling is not mixed
up with the service being down.
"""

import random

from locust import HttpUser, between, task

QUESTIONS = [
    "ما القانون الذي يحكم الالتزامات غير التعاقدية؟",
    "ما الذي يضمنه البائع للمشتري بالنسبة للانتفاع بالمبيع؟",
    "متى يكون شرط إسقاط ضمان العيب أو تقليله باطلًا؟",
    "What is the effect of force majeure on tort liability?",
    "Which calendar is used to calculate time limits?",
]

ASK_TIMEOUT_SECONDS = 120


class LegalRAGUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")

    @task(4)
    def ask(self):
        with self.client.post(
            "/ask",
            json={"question": random.choice(QUESTIONS), "top_k": 5},
            name="POST /ask",
            timeout=ASK_TIMEOUT_SECONDS,
            catch_response=True,
        ) as response:
            if response.status_code == 0:
                response.failure("connection failed or timed out")
            elif response.status_code == 429:
                response.failure("429 rate limited")
            elif response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            else:
                response.success()