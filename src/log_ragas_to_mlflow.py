
"""Log a completed RAGAS report to MLflow."""

import json
from pathlib import Path

import mlflow

from src.config import (
    CHUNK_STRATEGY,
    EMBEDDING_MODEL,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "data" / "evaluation" / "ragas_scores.json"

REQUIRED_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def main() -> None:
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"RAGAS report not found: {REPORT_PATH}")

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    if report.get("evaluation_complete") is not True:
        raise ValueError("Refusing to log an incomplete RAGAS evaluation.")

    total_questions = report.get("total_valid_questions")
    scored = report.get("successfully_scored_questions", {})
    metrics = report.get("metrics", {})

    if not isinstance(total_questions, int) or total_questions < 1:
        raise ValueError("Invalid total_valid_questions in RAGAS report.")

    for name in REQUIRED_METRICS:
        value = metrics.get(name)
        count = scored.get(name)

        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Missing or invalid metric: {name}")

        if count != total_questions:
            raise ValueError(
                f"{name} is incomplete: scored {count}/{total_questions}"
            )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("egyptian-legal-rag")

    with mlflow.start_run(run_name="ragas-baseline"):
        mlflow.set_tags(
            {
                "evaluation_type": "ragas",
                "evaluation_complete": "true",
                "source_file": str(report.get("source_file", "unknown")),
                "configured_experiment_name": MLFLOW_EXPERIMENT_NAME,
            }
        )

        mlflow.log_params(
            {
                "evaluator": str(report.get("evaluator", "unknown")),
                "total_questions": total_questions,
                "embedding_model": EMBEDDING_MODEL,
                "chunk_strategy": CHUNK_STRATEGY,
                "faithfulness_threshold": report.get(
                    "faithfulness_threshold", 0.75
                ),
            }
        )

        mlflow.log_metrics(
            {name: float(metrics[name]) for name in REQUIRED_METRICS}
        )

        mlflow.log_metric(
            "faithfulness_threshold_pass",
            float(
                metrics["faithfulness"]
                >= float(report.get("faithfulness_threshold", 0.75))
            ),
        )

        mlflow.log_artifact(str(REPORT_PATH), artifact_path="evaluation")

        print("MLflow run logged successfully.")
        print(f"Experiment: egyptian-legal-rag")
        print(f"Questions: {total_questions}")
        for name in REQUIRED_METRICS:
            print(f"{name}: {metrics[name]:.4f}")


if __name__ == "__main__":
    main()
