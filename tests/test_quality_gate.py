import json

import pytest

from src.quality_gate import decide, read_faithfulness


def test_decide_levels():
    assert decide(0.70, 0.75, 0.80) == 1
    assert decide(0.77, 0.75, 0.80) == 2
    assert decide(0.85, 0.75, 0.80) == 0
    assert decide(0.75, 0.75, 0.80) == 2


def test_read_faithfulness_rejects_incomplete_report(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"evaluation_complete": False}), encoding="utf-8")
    with pytest.raises(ValueError):
        read_faithfulness(path)


def test_read_faithfulness_reads_value(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(
        json.dumps(
            {
                "evaluation_complete": True,
                "total_valid_questions": 20,
                "metrics": {"faithfulness": 0.8},
            }
        ),
        encoding="utf-8",
    )
    assert read_faithfulness(path) == (0.8, 20)
