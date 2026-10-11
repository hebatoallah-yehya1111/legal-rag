
"""Validate the generated RAG evaluation results."""

import json
from pathlib import Path

RESULTS_PATH = Path("data/evaluation/rag_results.json")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}"
        )

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list):
        raise ValueError("Expected a JSON list of evaluation results.")

    total = len(results)
    successful = 0
    failed = 0
    empty_answers = 0
    empty_contexts = 0
    missing_ground_truth = 0
    missing_references = 0
    no_citations = 0

    print("=" * 65)
    print("RAG EVALUATION DATA VALIDATION")
    print("=" * 65)

    for item in results:
        answer = item.get("answer")
        contexts = item.get("retrieved_contexts") or []
        ground_truth = item.get("ground_truth")
        reference_text = item.get("reference_text")
        citations = item.get("citations") or []

        if item.get("error"):
            failed += 1
        elif isinstance(answer, str) and answer.strip():
            successful += 1

        if not isinstance(answer, str) or not answer.strip():
            empty_answers += 1

        if not contexts or not any(
            isinstance(context, str) and context.strip()
            for context in contexts
        ):
            empty_contexts += 1

        if not ground_truth:
            missing_ground_truth += 1

        if not reference_text:
            missing_references += 1

        if not citations:
            no_citations += 1

    print(f"Total questions            : {total}")
    print(f"Successful responses       : {successful}")
    print(f"Failed responses           : {failed}")
    print(f"Empty answers              : {empty_answers}")
    print(f"Empty retrieved contexts   : {empty_contexts}")
    print(f"Missing ground truth       : {missing_ground_truth}")
    print(f"Missing reference text     : {missing_references}")
    print(f"Responses without citations: {no_citations}")

    print("\nSample results")
    print("-" * 65)

    for item in results[:3]:
        print(f"\nID: {item.get('id')}")
        print(f"Question: {item.get('question')}")
        print(f"Answer: {str(item.get('answer'))[:250]}")
        print(f"Citations: {item.get('citations', [])}")
        print(
            "Retrieved contexts:",
            len(item.get("retrieved_contexts") or []),
        )

    print("\nValidation report completed.")


if __name__ == "__main__":
    main()
