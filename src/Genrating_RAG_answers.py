
"""Evaluate the Egyptian Civil Code RAG pipeline."""

import json
from pathlib import Path
from typing import Any

from src.rag import LegalRAG

DATASET_PATH = Path("data/evaluation/evaluation_dataset.json")
RESULTS_PATH = Path("data/evaluation/rag_results.json")


def main() -> None:
    # Load the evaluation dataset.
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("The evaluation dataset must be a JSON list.")

    # Configure the number of retrieved articles here.
    rag = LegalRAG(top_k=5)

    results: list[dict[str, Any]] = []

    for i, item in enumerate(dataset, start=1):
        question = item["question"]

        print(f"[{i}/{len(dataset)}] {question}")

        result: dict[str, Any] = {
            "id": item.get("id", i),
            "question": question,
            "language": item.get("language"),
            "ground_truth": item.get("ground_truth"),
            "reference_text": item.get("reference_text"),
            "scope": item.get("scope"),
        }

        try:
            # answer() performs retrieval and generation.
            response = rag.answer(question)

            # Retrieve article texts directly from the same
            # search results used to build the answer context.
            retrieved_articles = response.get(
                "retrieved_articles", []
            )

            retrieved_contexts = response.get(
                "retrieved_contexts"
            )

            # Backward-compatible fallback if retrieved_contexts
            # has not yet been added to LegalRAG.answer().
            if retrieved_contexts is None:
                retrieved, _ = rag.retrieve_context(question)
                retrieved_contexts = [
                    doc.get("document", "")
                    for doc in retrieved
                    if doc.get("document")
                ]

            result.update(
                {
                    "answer": response.get("answer"),
                    "citations": response.get("citations", []),
                    "retrieved_articles": retrieved_articles,
                    "retrieved_contexts": retrieved_contexts,
                    "model": response.get("model"),
                }
            )

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(f"ERROR: {error_message}")

            result.update(
                {
                    "answer": None,
                    "citations": [],
                    "retrieved_articles": [],
                    "retrieved_contexts": [],
                    "model": None,
                    "error": error_message,
                }
            )

        results.append(result)

    # Save results, including failed questions, for debugging.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    success_count = sum(
        1
        for result in results
        if result.get("answer") is not None
        and "error" not in result
    )

    failed_count = len(results) - success_count

    print("\n" + "=" * 60)
    print("RAG EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total questions : {len(results)}")
    print(f"Successful      : {success_count}")
    print(f"Failed          : {failed_count}")
    print(f"Results saved   : {RESULTS_PATH}")


if __name__ == "__main__":
    main()
