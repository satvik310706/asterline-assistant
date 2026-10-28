import json
from pathlib import Path

from asterline_assistant.reporting import publish_results, render_eval


def test_reporting_publishes_and_renders_generated_metrics(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    payload = {"provider": "fake", "config": "baseline", "cases": 1, "overall": {"fact_accuracy": {"value": 1.0, "applicable": 1, "total": 1}}, "by_category": {}, "failures": []}
    (results / "baseline_fake_20260101.json").write_text(json.dumps(payload), encoding="utf-8")
    final = tmp_path / "final"
    assert publish_results(results, final)
    output = tmp_path / "EVAL.md"
    render_eval(final, output)
    assert "1.0000" in output.read_text(encoding="utf-8")
