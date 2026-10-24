"""Measure retrieval quality with Hit@k and MRR. No LLM and no API quota needed.

For every in-scope question, the retriever returns articles, and we check at
which position the correct article (from gold_articles.json) appears.

Usage:
    python -m src.evaluate_retrieval
    python -m src.evaluate_retrieval --mlflow --run-name baseline --param max_chars=4000
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GOLD = ROOT / "data" / "evaluation" / "gold_articles.json"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation" / "retrieval_metrics.json"

KS = (1, 3, 5, 10)


def unique_in_order(numbers) -> list[int]:
    """Several chunks can come from one article, so keep each article once."""
    return list(dict.fromkeys(n for n in numbers if n is not None))


def first_rank(retrieved: list[int], gold: set[int]) -> int | None:
    """1-based position of the first correct article, or None if it is missing."""
    for position, number in enumerate(retrieved, start=1):
        if number in gold:
            return position
    return None


def summarize(ranks: list[int | None], ks=KS) -> dict[str, float]:
    total = len(ranks)
    if total == 0:
        return {}
    summary = {
        f"hit_at_{k}": sum(1 for r in ranks if r is not None and r <= k) / total
        for k in ks
    }
    summary["mrr"] = sum(1 / r for r in ranks if r) / total
    return summary


def evaluate(
    gold_rows: list[dict],
    search: Callable[[str, int], list[dict]],
    k_max: int = 10,
) -> list[dict]:
    """Run every in-scope question through `search(question, k)` and record the rank."""
    results = []
    for row in gold_rows:
        if row.get("scope") != "in_scope" or not row.get("gold_articles"):
            continue
        hits = search(row["question"], k_max)
        retrieved = unique_in_order(hit.get("article_number") for hit in hits)
        results.append(
            {
                "id": row["id"],
                "language": row.get("language"),
                "gold_articles": row["gold_articles"],
                "rank": first_rank(retrieved, set(row["gold_articles"])),
                "retrieved": retrieved,
            }
        )
    return results


def format_summary(title: str, summary: dict[str, float], count: int) -> str:
    cells = "  ".join(f"{name}={value:.3f}" for name, value in summary.items())
    return f"{title:<8} (n={count:>2})  {cells}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--mlflow", action="store_true", help="Log the run to MLflow.")
    parser.add_argument("--experiment", default="retrieval")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra setting to log, as key=value (for example max_chars=4000).",
    )
    args = parser.parse_args()

    # Heavy imports happen here, so the pure functions above stay easy to test.
    from src.config import EMBEDDING_MODEL
    from src.retrieval import Retriever

    gold_rows = json.loads(args.gold.read_text(encoding="utf-8"))
    retriever = Retriever()
    results = evaluate(
        gold_rows,
        lambda question, k: retriever.search(question, top_k=k),
        k_max=args.k_max,
    )

    overall = summarize([r["rank"] for r in results])
    by_language = {}
    for language in sorted({r["language"] for r in results if r["language"]}):
        ranks = [r["rank"] for r in results if r["language"] == language]
        by_language[language] = (summarize(ranks), len(ranks))

    print(format_summary("overall", overall, len(results)))
    for language, (summary, count) in by_language.items():
        print(format_summary(language, summary, count))

    misses = [r for r in results if r["rank"] is None]
    print(f"\nNot found in the top {args.k_max}: {len(misses)} question(s)")
    for miss in misses:
        print(f"  [{miss['id']}] gold={miss['gold_articles']} retrieved={miss['retrieved'][:5]}")

    report = {
        "embedding_model": EMBEDDING_MODEL,
        "k_max": args.k_max,
        "overall": overall,
        "by_language": {lang: summary for lang, (summary, _) in by_language.items()},
        "per_question": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out}")

    if args.mlflow:
        import mlflow

        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=args.run_name):
            params = {
                "embedding_model": EMBEDDING_MODEL,
                "k_max": args.k_max,
                "n_questions": len(results),
            }
            for item in args.param:
                key, _, value = item.partition("=")
                params[key.strip()] = value.strip()
            mlflow.log_params(params)
            mlflow.log_metrics(overall)
            for language, (summary, _) in by_language.items():
                mlflow.log_metrics({f"{language}_{k}": v for k, v in summary.items()})
            mlflow.log_artifact(str(args.out))
        print(f"Logged to MLflow experiment '{args.experiment}'.")


if __name__ == "__main__":
    main()