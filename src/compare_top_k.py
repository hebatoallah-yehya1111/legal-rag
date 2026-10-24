
"""Compare LegalRAG retrieval depths and log runs to MLflow."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mlflow

from src.config import (
    CHUNK_STRATEGY,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    MLFLOW_TRACKING_URI,
)
from src.rag import LegalRAG

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "evaluation" / "evaluation_dataset.json"
OUTPUT_DIR = ROOT / "data" / "evaluation" / "top_k"


def load_dataset() -> list[dict[str, Any]]:
    with DATASET.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    for index, item in enumerate(dataset, start=1):
        if not isinstance(item, dict) or not item.get("question"):
            raise ValueError(f"Invalid question record at index {index}.")

    return dataset


def run_experiment(dataset: list[dict[str, Any]], top_k: int) -> None:
    output_path = OUTPUT_DIR / f"rag_results_top_k_{top_k}.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize once per experiment.
    rag = LegalRAG(top_k=top_k)
    results: list[dict[str, Any]] = []
    start = time.perf_counter()

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        print(f"[top_k={top_k}] [{index}/{len(dataset)}] {question}")

        record = dict(item)
        try:
            response = rag.answer(question)
            record.update(
                {
                    "answer": response.get("answer"),
                    "citations": response.get("citations", []),
                    "retrieved_articles": response.get("retrieved_articles", []),
                    "retrieved_contexts": response.get("retrieved_contexts", []),
                    "model": response.get("model"),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "answer": None,
                    "citations": [],
                    "retrieved_articles": [],
                    "retrieved_contexts": [],
                    "model": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"ERROR: {record['error']}")

        results.append(record)

    elapsed = time.perf_counter() - start
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)

    successful = sum(
        1 for item in results
        if item.get("answer") is not None and "error" not in item
    )
    failed = len(results) - successful

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("egyptian-legal-rag")

    with mlflow.start_run(run_name=f"retrieval-top-k-{top_k}"):
        mlflow.log_params(
            {
                "top_k": top_k,
                "dataset": DATASET.name,
                "total_questions": len(dataset),
                "embedding_model": EMBEDDING_MODEL,
                "chunk_strategy": CHUNK_STRATEGY,
                "llm_model": GROQ_MODEL,
            }
        )
        mlflow.log_metrics(
            {
                "successful_answers": successful,
                "failed_answers": failed,
                "elapsed_seconds": elapsed,
            }
        )
        mlflow.set_tags(
            {
                "experiment_type": "top_k_comparison",
                "evaluation_stage": "answer_generation",
            }
        )
        mlflow.log_artifact(str(output_path), artifact_path="generation_results")

    print(
        f"Finished top_k={top_k}: {successful}/{len(results)} successful; "
        f"{elapsed:.1f}s; saved {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[3, 5, 8],
        help="Retrieval depths to compare (default: 3 5 8).",
    )
    args = parser.parse_args()

    if any(value < 1 for value in args.top_k):
        parser.error("All top-k values must be positive integers.")

    dataset = load_dataset()
    for top_k in args.top_k:
        run_experiment(dataset, top_k)


if __name__ == "__main__":
    main()
