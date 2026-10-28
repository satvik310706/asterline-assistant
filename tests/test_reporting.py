import json
from pathlib import Path

from asterline_assistant import observability
from asterline_assistant.reporting import publish_results, render_eval


def test_reporting_publishes_and_renders_generated_metrics(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    payload = {"status": "complete", "provider": "fake", "config": "baseline", "cases": 1, "overall": {"fact_accuracy": {"value": 1.0, "applicable": 1, "total": 1}}, "by_category": {}, "failures": []}
    (results / "baseline_fake_20260101.json").write_text(json.dumps(payload), encoding="utf-8")
    live_payload = {"status": "complete", "provider": "ollama", "model": "llama3.2:3b", "config": "baseline", "no_fallback": True, "cases": 1, "overall": {"fact_accuracy": {"value": 0.5, "applicable": 1, "total": 1}, "provider_error_count": 2}, "by_category": {}, "failures": []}
    (results / "baseline_ollama_20260101.json").write_text(json.dumps(live_payload), encoding="utf-8")
    final = tmp_path / "final"
    published = publish_results(results, final)
    assert len(published) == 2
    output = tmp_path / "EVAL.md"
    render_eval(final, output)
    rendered = output.read_text(encoding="utf-8")
    assert "1.0000" in rendered
    assert "not estimates of Gemini or Ollama answer quality" in rendered
    assert "baseline_ollama.json" in rendered
    assert "fallback disabled: `true`" in rendered
    assert "| provider_error_count | 2.0000 | count |" in rendered


def test_partial_results_are_not_published_as_submission_headlines(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "baseline_ollama_20261028T000000Z.json").write_text(
        json.dumps({"status": "complete", "provider": "ollama", "cases": 80}),
        encoding="utf-8",
    )
    (results / "baseline_ollama_20261029T000000Z.json").write_text(
        json.dumps({"status": "partial", "provider": "ollama", "cases": 80, "completed": 5}),
        encoding="utf-8",
    )
    final = tmp_path / "final"

    assert publish_results(results, final) == [final / "baseline_ollama.json"]
    report = json.loads((final / "baseline_ollama.json").read_text(encoding="utf-8"))
    assert report["status"] == "complete"


def test_limited_memory_eval_cannot_replace_full_submission_report(tmp_path: Path) -> None:
    results = tmp_path / "eval" / "results"
    datasets = tmp_path / "eval" / "datasets"
    results.mkdir(parents=True)
    datasets.mkdir(parents=True)
    (datasets / "research.jsonl").write_text('{"id":"r1"}\n', encoding="utf-8")
    (datasets / "memory.jsonl").write_text('{"id":"m1"}\n{"id":"m2"}\n', encoding="utf-8")
    (results / "memory_fake_20260101.json").write_text(
        json.dumps({"status": "complete", "provider": "fake", "scenarios": [{"id": "m1"}, {"id": "m2"}]}),
        encoding="utf-8",
    )
    (results / "memory_fake_20260102.json").write_text(
        json.dumps({"status": "complete", "provider": "fake", "scenarios": [{"id": "m1"}]}),
        encoding="utf-8",
    )

    final = tmp_path / "eval" / "final"
    assert publish_results(results, final) == [final / "memory_fake.json"]
    report = json.loads((final / "memory_fake.json").read_text(encoding="utf-8"))
    assert len(report["scenarios"]) == 2


def test_eval_renderer_computes_only_provider_matched_pair_deltas(tmp_path: Path) -> None:
    final = tmp_path / "final"
    final.mkdir()
    for config, score in (("baseline", 0.4), ("improved", 0.8)):
        payload = {
            "status": "complete",
            "provider": "ollama",
            "model": "llama3.2:3b",
            "config": config,
            "pipeline_revision": "test-v1",
            "cases": 1,
            "overall": {},
            "by_category": {"arithmetic": {"fact_accuracy": {"value": score, "applicable": 10, "total": 13}}},
            "failures": [],
        }
        (final / f"{config}_ollama.json").write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "EVAL.md"
    render_eval(final, output)
    rendered = output.read_text(encoding="utf-8")
    assert "ollama/llama3.2:3b | arithmetic | fact_accuracy | 0.4000 (10) | 0.8000 (10) | +40.00 pp" in rendered


def test_memory_report_uses_scenario_schema_instead_of_zero_cases(tmp_path: Path) -> None:
    final = tmp_path / "final"
    final.mkdir()
    payload = {
        "status": "complete",
        "provider": "fake",
        "scenarios": [
            {"id": f"M{index:02}", "category": "isolation" if index < 2 else "forget", "passed": True}
            for index in range(34)
        ],
        "category_pass_rates": {"isolation": 1.0, "forget": 1.0},
        "memory_write_precision": 1.0,
        "memory_write_recall": 1.0,
        "forget_success": 1.0,
        "isolation_leakage": 0,
        "secret_leak_count": 0,
        "sensitive_stored_count": 0,
    }
    (final / "memory_fake.json").write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "EVAL.md"
    render_eval(final, output)
    rendered = output.read_text(encoding="utf-8")

    assert "- scenarios: 34" in rendered
    assert "- passed: 34/34" in rendered
    assert "cases: 0" not in rendered
    assert "| isolation | 1.0000 | 2/2 |" in rendered
    assert "| memory_write_precision | 1.0000 | aggregate over 34 scenarios |" in rendered


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


def test_langsmith_upload_includes_report_cases_and_redacts_sensitive_text(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "research.jsonl"
    dataset.write_text(
        json.dumps({"id": "S01", "category": "single-fact", "question": "Use password=synthetic-pass-77", "expected": "answer"}) + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "improved_fake.json"
    report.write_text(json.dumps({
        "config": "improved",
        "provider": "fake",
        "case_results": [{"id": "S01", "question": "Use password=synthetic-pass-77", "answer": "Authorization: Bearer synthetic-token-88", "status": "ok"}],
    }), encoding="utf-8")
    captured: list[dict] = []

    class Dataset:
        id = "dataset-id"

    class Client:
        def create_dataset(self, **kwargs):
            return Dataset()

        def create_examples(self, **kwargs):
            captured.extend(kwargs["examples"])

    monkeypatch.setenv("LANGSMITH_API_KEY", "present")
    monkeypatch.setattr("langsmith.Client", Client)

    message = observability.maybe_upload(dataset, report)

    assert "attempted" in message
    assert len(captured) == 1
    encoded = json.dumps(captured)
    assert "synthetic-pass-77" not in encoded
    assert "synthetic-token-88" not in encoded
    assert "[REDACTED]" in encoded
    assert captured[0]["outputs"]["answer"] == "Authorization: Bearer [REDACTED]"


def test_trace_handoff_marks_missing_real_links_and_explains_public_sharing() -> None:
    docs = Path(__file__).parents[1] / "docs" / "TRACES.md"
    contents = docs.read_text(encoding="utf-8")

    assert "PENDING — paste public trace URL" in contents
    assert "local illustrations, not remotely captured traces" in contents
    assert "radialpulse@nxtwave.co.in" in contents
    assert "More` menu and choose `Share`" in contents
