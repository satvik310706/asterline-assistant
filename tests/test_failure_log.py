import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.generate_failure_log import render


def test_failure_log_is_data_driven_and_reports_timeout_before_after(tmp_path: Path) -> None:
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    dataset = tmp_path / "research.jsonl"
    common = {"provider": "ollama", "model": "llama3.2:3b", "config": "baseline"}
    before.write_text(
        json.dumps(common | {
            "overall": {"fact_accuracy": {"value": 0.5, "applicable": 2, "total": 2}, "provider_error_count": 1},
            "case_results": [
                {"id": "A01", "question": "What is fact A?", "status": "provider_error", "error_class": "TimeoutError", "checks": {}, "answer": "source only"},
                {"id": "A02", "question": "What is fact B?", "status": "ok", "checks": {"fact_accuracy": False}, "answer": "wrong"},
                {"id": "A03", "category": "arithmetic", "question": "What is sum C?", "status": "ok", "checks": {"fact_accuracy": False}, "answer": "not calculated"},
            ],
        }),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(common | {
            "overall": {"fact_accuracy": {"value": 0.5, "applicable": 2, "total": 2}, "provider_error_count": 0},
            "case_results": [
                {"id": "A01", "question": "What is fact A?", "status": "ok", "checks": {"fact_accuracy": True}, "answer": "fact A"},
                {"id": "A02", "question": "What is fact B?", "status": "ok", "checks": {"fact_accuracy": False}, "answer": "wrong"},
                {"id": "A03", "category": "arithmetic", "question": "What is sum C?", "status": "ok", "checks": {"fact_accuracy": True}, "answer": "sum C"},
            ],
        }),
        encoding="utf-8",
    )
    dataset.write_text(
        '{"id":"A01","question":"What is fact A?","expected":"answer"}\n'
        '{"id":"A02","question":"What is fact B?","expected":"answer"}\n'
        '{"id":"A03","question":"What is sum C?","expected":"answer"}\n',
        encoding="utf-8",
    )

    output = render(before, after, dataset)

    assert "provider_error_count | 1 | 0" in output
    assert "previous 12-second socket timeout" in output
    assert "Raised the local request timeout to 45 seconds" in output
    assert "| A02 | What is fact B? | answer |" in output
    assert "same case later completed" in output
    assert "| A03 | What is sum C? | answer |" in output
    assert "deterministic evidence-backed calculator" in output
