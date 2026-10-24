"""Quality gate for RAGAS faithfulness.

Exit codes:
    0  faithfulness is at or above the alert level
    1  faithfulness is below the fail level (CI must fail)
    2  faithfulness passes the fail level but is below the alert level (alert sent)

Usage:
    python -m src.quality_gate
    python -m src.quality_gate --fail-below 0.75 --alert-below 0.80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = ROOT / "data" / "evaluation" / "ragas_scores.json"


def read_faithfulness(scores_path: Path, exclude_out_of_scope: bool = False) -> tuple[float, int]:
    """Return (faithfulness, number_of_questions) from a RAGAS report."""
    report = json.loads(scores_path.read_text(encoding="utf-8"))

    if report.get("evaluation_complete") is not True:
        raise ValueError("The RAGAS report is incomplete. Re-run the evaluation.")

    if not exclude_out_of_scope:
        value = report["metrics"]["faithfulness"]
        return float(value), int(report["total_valid_questions"])

    rows = [
        row for row in report["per_question"]
        if not row.get("out_of_scope") and row.get("faithfulness") is not None
    ]
    if not rows:
        raise ValueError("No scored in-scope questions found.")
    return sum(row["faithfulness"] for row in rows) / len(rows), len(rows)


def decide(faithfulness: float, fail_below: float, alert_below: float) -> int:
    """Pure decision function, easy to unit test."""
    if faithfulness < fail_below:
        return 1
    if faithfulness < alert_below:
        return 2
    return 0


def send_alert(message: str) -> bool:
    """Post a message to a Slack/Discord-style webhook if ALERT_WEBHOOK_URL is set."""
    url = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    print(f"ALERT: {message}")
    if not url:
        print("ALERT_WEBHOOK_URL is not set, so the alert was only printed.")
        return False

    body = json.dumps({"text": message, "content": message}).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10):
            return True
    except OSError as exc:  # network problems must not hide the gate result
        print(f"Could not send the alert: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--fail-below", type=float, default=0.75)
    parser.add_argument("--alert-below", type=float, default=0.80)
    args = parser.parse_args()

    value, count = read_faithfulness(args.scores)
    code = decide(value, args.fail_below, args.alert_below)

    print(f"faithfulness={value:.3f} on {count} questions "
          f"(fail below {args.fail_below}, alert below {args.alert_below})")

    if code == 1:
        send_alert(f"Legal RAG faithfulness {value:.3f} is below the fail level {args.fail_below}.")
        print("GATE FAILED")
    elif code == 2:
        send_alert(f"Legal RAG faithfulness {value:.3f} is below the alert level {args.alert_below}.")
        print("GATE PASSED with an alert")
    else:
        print("GATE PASSED")
    return code


if __name__ == "__main__":
    sys.exit(main())
