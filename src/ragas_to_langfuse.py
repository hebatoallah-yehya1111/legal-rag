"""Send RAGAS faithfulness scores to Langfuse, attached to a trace per question.

For each question in rag_results.json this creates a Langfuse trace (with the
question, answer and cited articles) and attaches that question's RAGAS
faithfulness score to it, so low-faithfulness answers can be found in the
Langfuse UI.

Needs: pip install -e ".[observability]" and LANGFUSE_PUBLIC_KEY,
LANGFUSE_SECRET_KEY, LANGFUSE_HOST in .env (self-hosted Langfuse).

Usage:
    python -m src.ragas_to_langfuse
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "evaluation" / "rag_results.json"
SCORES = ROOT / "data" / "evaluation" / "ragas_scores.json"


def main() -> None:
    from langfuse import get_client

    import src.config  # noqa: F401  (loads .env so the Langfuse keys are available)

    langfuse = get_client()
    results = {r["question"]: r for r in json.loads(RESULTS.read_text(encoding="utf-8"))}
    scores = json.loads(SCORES.read_text(encoding="utf-8"))["per_question"]

    sent = 0
    for row in scores:
        record = results.get(row["question"])
        if not record or row.get("faithfulness") is None:
            continue

        with langfuse.start_as_current_span(name="legal-rag.ragas-eval") as span:
            span.update(
                input={"question": record["question"]},
                output={
                    "answer": record.get("answer"),
                    "sources": record.get("citations", []),
                },
                metadata={"retrieved": [a.get("article_number") for a in record.get("retrieved_articles", [])]},
            )
            span.score_trace(name="ragas_faithfulness", value=float(row["faithfulness"]))
            span.score_trace(name="ragas_answer_relevancy", value=float(row["answer_relevancy"]))
        sent += 1

    langfuse.flush()
    print(f"Sent {sent} traces with faithfulness scores to Langfuse.")


if __name__ == "__main__":
    main()
