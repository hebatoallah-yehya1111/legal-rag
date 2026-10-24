"""Guard the required minimum evaluation-set size and schema."""
import json
from pathlib import Path


def test_evaluation_dataset_has_at_least_50_unique_questions():
    path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "evaluation_dataset.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    assert len(dataset) >= 50
    assert len({row["id"] for row in dataset}) == len(dataset)
    for row in dataset:
        assert row.get("question", "").strip()
        assert row.get("ground_truth", "").strip()
        assert row.get("scope") in {"in_scope", "out_of_scope"}
