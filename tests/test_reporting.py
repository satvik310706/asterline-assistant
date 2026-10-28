import json
from pathlib import Path

from asterline_assistant import observability
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
    rendered = output.read_text(encoding="utf-8")
    assert "1.0000" in rendered
    assert "not estimates of Gemini or Ollama answer quality" in rendered


def test_langsmith_upload_uses_current_client_api(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "research.jsonl"
    dataset.write_text('{"id":"x","question":"q","expected":"answer"}\n', encoding="utf-8")
    calls: list[tuple[str, object]] = []

    class Dataset:
        id = "dataset-id"

    class Client:
        def create_dataset(self, **kwargs):
            calls.append(("dataset", kwargs))
            return Dataset()

        def create_examples(self, **kwargs):
            calls.append(("examples", kwargs))

    monkeypatch.setenv("LANGSMITH_API_KEY", "present")
    monkeypatch.setattr("langsmith.Client", Client)
    message = observability.maybe_upload(dataset)
    assert "attempted" in message
    assert [kind for kind, _ in calls] == ["dataset", "examples"]
    assert calls[1][1]["examples"][0]["metadata"]["synthetic"] is True
