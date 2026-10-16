"""Find the gold article number for every in-scope evaluation question.

Each question has a reference_text (an English quote from the right article).
This script finds which article contains that quote, so retrieval can be
measured with Hit@k and MRR instead of guessing.

Usage:
    python -m src.build_gold_articles
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUESTIONS = ROOT / "data" / "evaluation" / "rag_results.json"
DEFAULT_ARTICLES = ROOT / "data" / "processed" / "articles.json"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation" / "gold_articles.json"
DEFAULT_OVERRIDES = ROOT / "data" / "evaluation" / "gold_overrides.json"

PARTIAL_THRESHOLD = 0.5


def normalize(text: str | None) -> str:
    """Lowercase and keep only letters and digits, so punctuation never blocks a match."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def shingles(words: list[str], size: int = 4) -> set[tuple[str, ...]]:
    size = max(1, min(size, len(words)))
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def match_score(reference: str, article_text: str) -> float:
    """1.0 when the quote is inside the article, otherwise the share of overlapping 4-word runs."""
    if not reference or not article_text:
        return 0.0
    if reference in article_text:
        return 1.0
    ref_shingles = shingles(reference.split())
    if not ref_shingles:
        return 0.0
    text_shingles = shingles(article_text.split())
    return len(ref_shingles & text_shingles) / len(ref_shingles)


def find_gold(reference_text: str, articles: list[dict]) -> tuple[list[int], float, str]:
    reference = normalize(reference_text)
    scored = []
    for article in articles:
        if article.get("is_repealed"):
            continue
        score = match_score(reference, normalize(article.get("text_en")))
        if score > 0:
            scored.append((score, article["article_number"]))

    if not scored:
        return [], 0.0, "none"

    best = max(score for score, _ in scored)
    if best < PARTIAL_THRESHOLD:
        return [], best, "none"

    candidates = sorted(number for score, number in scored if score == best)
    if best == 1.0:
        status = "exact" if len(candidates) == 1 else "ambiguous"
    else:
        status = "partial"
    return candidates, best, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--articles", type=Path, default=DEFAULT_ARTICLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    articles = json.loads(args.articles.read_text(encoding="utf-8"))
    by_number = {a["article_number"]: a for a in articles}

    # Answers decided by hand: {"q003": [23], ...}. They win over the automatic match.
    overrides: dict[str, list[int]] = {}
    if args.overrides.exists():
        overrides = json.loads(args.overrides.read_text(encoding="utf-8"))

    gold = []
    for item in questions:
        if item.get("scope") != "in_scope":
            gold.append(
                {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "language": item.get("language"),
                    "scope": item.get("scope"),
                    "gold_articles": [],
                    "match_status": "out_of_scope",
                    "match_score": None,
                }
            )
            continue

        numbers, score, status = find_gold(item.get("reference_text", ""), articles)
        if item.get("id") in overrides:
            numbers, status = overrides[item["id"]], "manual"
        gold.append(
            {
                "id": item.get("id"),
                "question": item.get("question"),
                "language": item.get("language"),
                "scope": item.get("scope"),
                "gold_articles": numbers,
                "match_status": status,
                "match_score": round(score, 3),
                "reference_text": item.get("reference_text"),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for row in gold:
        counts[row["match_status"]] = counts.get(row["match_status"], 0) + 1
    print("Matches by status:", counts)
    print(f"Saved: {args.out}\n")

    # Only ASCII is printed, so the Windows console never chokes on Arabic.
    # "partial" means the reference was a paraphrase of one clear article, so it is accepted.
    needs_review = [r for r in gold if r["match_status"] in {"ambiguous", "none"}]
    if not needs_review:
        print("Nothing left to check by hand.")
        return

    print("CHECK THESE BY HAND:")
    for row in needs_review:
        print(f"\n[{row['id']}] status={row['match_status']} score={row['match_score']}")
        print("  reference:", (row["reference_text"] or "")[:140])
        for number in row["gold_articles"][:3]:
            print(f"  candidate Article {number}:", (by_number[number]["text_en"] or "")[:140])


if __name__ == "__main__":
    main()