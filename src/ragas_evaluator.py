"""Evaluate the Arabic legal RAG pipeline using RAGAS with a Groq judge.

Features:
- One metric per question, with checkpointing after every successful score.
- Resume safely: checkpoint keys include the complete evaluation input.
- Rate limiting and retry handling for transient quota errors.
- Retry on malformed JSON from the evaluator model.
- Separate partial results from complete evaluation results.
- Optional CI quality gate for faithfulness.

Environment:
    GROQ_API_KEY, RAGAS_GROQ_MODEL, RAGAS_RPM
    RAGAS_MAX_OUTPUT_TOKENS
    RAGAS_MAX_QUESTIONS
    RAGAS_METRICS
    RAGAS_MAX_CONTEXTS
    RAGAS_MAX_CONTEXT_CHARS
    RAGAS_FAIL_ON_THRESHOLD=true  # Enable the CI quality gate
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from src.config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    PROJECT_ROOT,
    RAGAS_FAITHFULNESS_THRESHOLD,
)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RESULTS_PATH = Path(
    os.getenv(
        "RAGAS_RESULTS_PATH",
        str(PROJECT_ROOT / "data" / "evaluation" / "rag_results.json"),
    )
)

OUTPUT_PATH = Path(
    os.getenv(
        "RAGAS_OUTPUT_PATH",
        str(PROJECT_ROOT / "data" / "evaluation" / "ragas_scores.json"),
    )
)

CHECKPOINT_PATH = OUTPUT_PATH.with_suffix(".checkpoint.json")

# The judge model. Keep it different from the generator model (GROQ_MODEL).
RAGAS_GROQ_MODEL = os.getenv("RAGAS_GROQ_MODEL", "openai/gpt-oss-120b")
RAGAS_RPM = float(os.getenv("RAGAS_RPM", "8"))

# Arabic text escaped as \uXXXX inside JSON uses many tokens, so keep this high.
MAX_OUTPUT_TOKENS = int(os.getenv("RAGAS_MAX_OUTPUT_TOKENS", "8192"))

MAX_QUESTIONS = int(os.getenv("RAGAS_MAX_QUESTIONS", "0"))
MAX_CONTEXTS = int(os.getenv("RAGAS_MAX_CONTEXTS", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("RAGAS_MAX_CONTEXT_CHARS", "1200"))

FAIL_ON_THRESHOLD = (
    os.getenv("RAGAS_FAIL_ON_THRESHOLD", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

ALL_METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

METRIC_NAMES = [
    name.strip()
    for name in os.getenv(
        "RAGAS_METRICS", ",".join(ALL_METRICS)
    ).split(",")
    if name.strip() in ALL_METRICS
]

# Use one candidate to reduce unnecessary API usage.
answer_relevancy.strictness = 1


class DailyQuotaExceeded(Exception):
    """Raised when the API quota is exhausted and waiting is not useful."""


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def row_key(row: dict[str, Any]) -> str:
    """Build a stable key from every field that affects evaluation."""

    payload = json.dumps(
        {
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
            "ground_truth": row["ground_truth"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_results() -> list[dict[str, Any]]:
    """Load and validate evaluation examples."""

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_PATH}"
        )

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if not isinstance(results, list) or not results:
        raise ValueError(
            "Expected a non-empty JSON list in the results file."
        )

    valid_rows: list[dict[str, Any]] = []
    skipped = 0

    for item in results:
        if not isinstance(item, dict) or item.get("error"):
            skipped += 1
            continue

        question = item.get("question")
        answer = item.get("answer")
        contexts = item.get("retrieved_contexts")
        ground_truth = item.get("ground_truth")

        if not all(
            isinstance(value, str) and value.strip()
            for value in (question, answer, ground_truth)
        ):
            skipped += 1
            continue

        if not isinstance(contexts, list):
            skipped += 1
            continue

        clean_contexts = [
            context.strip()[:MAX_CONTEXT_CHARS]
            for context in contexts
            if isinstance(context, str) and context.strip()
        ][:MAX_CONTEXTS]

        if not clean_contexts:
            skipped += 1
            continue

        valid_rows.append(
            {
                "question": question.strip(),
                "answer": answer.strip(),
                "contexts": clean_contexts,
                "ground_truth": ground_truth.strip(),
            }
        )

    print(f"Source records: {len(results)}")
    print(f"Valid questions: {len(valid_rows)}")
    print(f"Skipped records: {skipped}")

    if not valid_rows:
        raise ValueError("No valid evaluation rows were found.")

    if MAX_QUESTIONS > 0:
        valid_rows = valid_rows[:MAX_QUESTIONS]
        print(f"TEST MODE: evaluating {len(valid_rows)} questions.")

    return valid_rows


# ---------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------

def load_checkpoint() -> dict[str, dict[str, float]]:
    """Load valid checkpoint entries; invalid files start fresh."""

    if not CHECKPOINT_PATH.exists():
        return {}

    try:
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Checkpoint root must be an object.")

        checkpoint: dict[str, dict[str, float]] = {}

        for key, scores in data.items():
            if not isinstance(key, str) or not isinstance(scores, dict):
                continue

            valid_scores = {}
            for metric_name, value in scores.items():
                if (
                    metric_name in ALL_METRICS
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                ):
                    valid_scores[metric_name] = float(value)

            checkpoint[key] = valid_scores

        return checkpoint

    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(
            f"Warning: checkpoint could not be loaded ({exc}). "
            "Starting with an empty checkpoint."
        )
        return {}


def save_checkpoint(
    checkpoint: dict[str, dict[str, float]],
) -> None:
    """Write atomically so interrupted runs preserve previous progress."""

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")

    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(
            checkpoint,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    tmp_path.replace(CHECKPOINT_PATH)


# ---------------------------------------------------------------------
# Evaluator models
# ---------------------------------------------------------------------

def build_evaluator_llm() -> LangchainLLMWrapper:
    """Build the RAGAS judge on Groq."""

    if not GROQ_API_KEY:
        raise OSError("GROQ_API_KEY is missing. Set it in .env.")
    if RAGAS_RPM <= 0:
        raise ValueError("RAGAS_RPM must be greater than zero.")

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise OSError(
            'Groq judge support is missing. Install with: pip install -e ".[evaluation]"'
        ) from exc

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=RAGAS_RPM / 60.0,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )

    print(
        f"Evaluator LLM: Groq / {RAGAS_GROQ_MODEL} "
        f"(limit {RAGAS_RPM} RPM, max output {MAX_OUTPUT_TOKENS} tokens)"
    )
    llm = ChatGroq(
        model=RAGAS_GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=180,
        max_retries=1,
        rate_limiter=rate_limiter,
    )
    return LangchainLLMWrapper(llm)


def build_evaluator_embeddings() -> LangchainEmbeddingsWrapper:
    print(f"Evaluator embeddings: {EMBEDDING_MODEL} (local CPU)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return LangchainEmbeddingsWrapper(embeddings)


# ---------------------------------------------------------------------
# Scoring and retries
# ---------------------------------------------------------------------

def classify_error(exc: Exception) -> str:
    """Classify quota failures as daily, minute, or other."""

    text = f"{type(exc).__name__} {exc}".lower()
    compact = (
        text.replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    quota_markers = (
        "429",
        "resourceexhausted",
        "quota",
        "ratelimit",
        "too many requests",
    )

    if not any(marker in compact for marker in quota_markers):
        return "other"

    daily_markers = (
        "perday",
        "perdayperproject",
        "daily",
        "dayquota",
        "quotaexceededforday",
        "tokensperday",
        "requestsperday",
    )

    if any(marker in compact for marker in daily_markers):
        return "daily"

    return "minute"


def score_one(
    metric_name: str,
    row: dict[str, Any],
    llm: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
    run_config: RunConfig,
) -> float | None:
    """Evaluate one metric on one example."""

    result = evaluate(
        dataset=Dataset.from_list([row]),
        metrics=[ALL_METRICS[metric_name]],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=True,
        show_progress=False,
    )

    df = result.to_pandas()

    if metric_name not in df.columns or df.empty:
        return None

    value = df[metric_name].iloc[0]

    if value is None:
        return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(score):
        return None

    return score


def score_with_retry(
    metric_name: str,
    row: dict[str, Any],
    llm: LangchainLLMWrapper,
    embeddings: LangchainEmbeddingsWrapper,
    run_config: RunConfig,
) -> float | None:
    """Retry rate-limit and malformed-JSON errors.

    Configuration errors are not retried.
    """

    for attempt in range(3):
        try:
            return score_one(
                metric_name, row, llm, embeddings, run_config
            )

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            kind = classify_error(exc)

            if kind == "daily":
                raise DailyQuotaExceeded(str(exc)) from exc

            if kind == "minute":
                if attempt < 2:
                    print("    rate limit hit; waiting 65 seconds...")
                    time.sleep(65)
                    continue

                raise DailyQuotaExceeded(
                    "Rate-limit errors persisted after retries. "
                    "Evaluation stopped safely; retry later."
                ) from exc

            if type(exc).__name__ in {
                "NotFound",
                "PermissionDenied",
                "Unauthenticated",
                "InvalidArgument",
                "AuthenticationError",
                "BadRequestError",
            }:
                raise RuntimeError(
                    f"Evaluator configuration error ({type(exc).__name__}): "
                    f"{str(exc)[:400]}\n"
                    "Check RAGAS_GROQ_MODEL, GROQ_API_KEY, and account access."
                ) from exc

            if type(exc).__name__ == "OutputParserException":
                if attempt < 2:
                    print(
                        "    malformed JSON from model; "
                        f"retrying ({attempt + 1}/2)..."
                    )
                    continue

                print("    parser error persisted; skipping this question")
                return None

            print(
                f"    failed ({type(exc).__name__}): "
                f"{str(exc)[:200]}"
            )
            return None

    return None


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def build_report(
    rows: list[dict[str, Any]],
    checkpoint: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Build a report using only the current evaluation inputs."""

    per_question = []
    values_by_metric: dict[str, list[float]] = {
        name: [] for name in METRIC_NAMES
    }

    for row in rows:
        scores = checkpoint.get(row_key(row), {})

        question_result = {"question": row["question"]}

        for metric_name in METRIC_NAMES:
            score = scores.get(metric_name)
            question_result[metric_name] = score

            if score is not None and math.isfinite(score):
                values_by_metric[metric_name].append(score)

        per_question.append(question_result)

    summary = {
        name: (
            sum(values) / len(values)
            if values else None
        )
        for name, values in values_by_metric.items()
    }

    scored_counts = {
        name: len(values)
        for name, values in values_by_metric.items()
    }

    complete_by_metric = {
        name: scored_counts[name] == len(rows)
        for name in METRIC_NAMES
    }

    return {
        "evaluator": {
            "provider": "groq",
            "model": RAGAS_GROQ_MODEL,
            "embedding_model": EMBEDDING_MODEL,
        },
        "source_file": str(RESULTS_PATH),
        "total_valid_questions": len(rows),
        "successfully_scored_questions": scored_counts,
        "complete_by_metric": complete_by_metric,
        "evaluation_complete": all(complete_by_metric.values()),
        "faithfulness_threshold": RAGAS_FAITHFULNESS_THRESHOLD,
        "metrics": summary,
        "per_question": per_question,
    }


def print_summary(
    report: dict[str, Any],
    total: int,
) -> None:
    print("\n" + "=" * 60)
    print("RAGAS Evaluation Summary")
    print("=" * 60)

    for metric_name in METRIC_NAMES:
        score = report["metrics"][metric_name]
        scored = report["successfully_scored_questions"][metric_name]
        status = "COMPLETE" if scored == total else "PARTIAL"

        if score is None:
            print(
                f"{metric_name:20s}: No score "
                f"({scored}/{total}) [{status}]"
            )
        else:
            print(
                f"{metric_name:20s}: {score:.4f} "
                f"({scored}/{total}) [{status}]"
            )

    if "faithfulness" not in METRIC_NAMES:
        print("\nFaithfulness check: NOT RUN")
        return

    threshold = RAGAS_FAITHFULNESS_THRESHOLD
    score = report["metrics"]["faithfulness"]
    scored = report["successfully_scored_questions"]["faithfulness"]

    print(f"\nFaithfulness threshold: {threshold:.2f}")

    if score is None:
        print("Faithfulness check: INCONCLUSIVE")
    elif scored < total:
        print(
            f"Faithfulness check: PARTIAL ({scored}/{total}); "
            f"current mean is {'above' if score >= threshold else 'below'} "
            "the threshold."
        )
    elif score >= threshold:
        print("Faithfulness check: PASS")
    else:
        print("Faithfulness check: BELOW THRESHOLD")


def write_report(report: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH.with_suffix(".tmp")

    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    tmp_path.replace(OUTPUT_PATH)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Arabic Legal RAG - RAGAS Evaluation")
    print("=" * 60)

    if not METRIC_NAMES:
        raise ValueError(
            "No valid metrics selected. Check the RAGAS_METRICS setting."
        )

    rows = load_results()
    checkpoint = load_checkpoint()

    llm = build_evaluator_llm()
    embeddings = build_evaluator_embeddings()

    run_config = RunConfig(
        timeout=180,
        max_retries=2,
        max_wait=30,
        max_workers=1,
    )

    quota_hit = False

    # Complete faithfulness first before spending quota on other metrics.
    for metric_name in METRIC_NAMES:
        todo = [
            row for row in rows
            if metric_name not in checkpoint.get(row_key(row), {})
        ]

        print(
            f"\n[{metric_name}] {len(todo)} to score, "
            f"{len(rows) - len(todo)} already done."
        )

        for index, row in enumerate(todo, start=1):
            print(f"  ({index}/{len(todo)}) scoring...")

            try:
                value = score_with_retry(
                    metric_name,
                    row,
                    llm,
                    embeddings,
                    run_config,
                )

            except DailyQuotaExceeded as exc:
                print(f"\nQuota/rate limit stopped evaluation: {str(exc)[:300]}")
                quota_hit = True
                break

            if value is None:
                print("    no score; this question can be retried later")
                continue

            key = row_key(row)
            checkpoint.setdefault(key, {})[metric_name] = value
            save_checkpoint(checkpoint)

            print(f"    {metric_name} = {value:.3f}")

        if quota_hit:
            break

    report = build_report(rows, checkpoint)
    write_report(report)
    print_summary(report, len(rows))

    if quota_hit:
        print(
            "\nEvaluation paused. Progress is saved in:\n"
            f"  {CHECKPOINT_PATH}\n"
            "Run again later to resume missing scores."
        )

    print(f"\nReport saved to: {OUTPUT_PATH}")

    # Optional strict mode for CI. Local evaluation can remain partial.
    if FAIL_ON_THRESHOLD:
        if "faithfulness" not in METRIC_NAMES:
            print(
                "CI QUALITY GATE FAILED: faithfulness was not selected.",
                file=sys.stderr,
            )
            return 1

        score = report["metrics"]["faithfulness"]
        scored = report["successfully_scored_questions"]["faithfulness"]

        if quota_hit or scored != len(rows) or score is None:
            print(
                "CI QUALITY GATE FAILED: faithfulness evaluation is "
                f"incomplete ({scored}/{len(rows)} scored).",
                file=sys.stderr,
            )
            return 1

        if score < RAGAS_FAITHFULNESS_THRESHOLD:
            print(
                "CI QUALITY GATE FAILED: faithfulness "
                f"{score:.4f} < {RAGAS_FAITHFULNESS_THRESHOLD:.4f}.",
                file=sys.stderr,
            )
            return 1

        print("CI QUALITY GATE PASSED.")

    print("Evaluation finished.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(
            "\nInterrupted. Successfully saved scores remain in the checkpoint.",
            file=sys.stderr,
        )
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nRAGAS evaluation failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise