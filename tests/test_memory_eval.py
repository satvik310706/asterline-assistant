import json
from pathlib import Path

from asterline_assistant.config import Settings
from asterline_assistant.memory_eval import _safe_report_text, run_memory_eval
from asterline_assistant.memory import Clock, MemoryStore, contains_sensitive
from datetime import datetime, timezone


def test_memory_eval_redacts_secret_like_values_from_report_fields() -> None:
    value = "sk-" + "test-FAKE1234567890abcdef"
    safe = _safe_report_text(f"memory_lacks:alice:{value}")

    assert "[REDACTED]" in safe
    assert not contains_sensitive(safe)


def test_memory_dataset_fake_has_no_isolation_or_secret_leaks(tmp_path: Path, monkeypatch) -> None:
    root = Path(__file__).parents[1]
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        corpus_path=root / "data/corpus.jsonl",
        chroma_path=tmp_path / "chroma",
        fastembed_cache_path=root / ".cache/fastembed",
        sqlite_path=tmp_path / "memory.sqlite3",
        llm_provider="fake",
    )
    result_path = run_memory_eval(root / "eval/datasets/memory.jsonl", settings, provider_name="fake")
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["isolation_leakage"] == 0
    assert report["secret_leak_count"] == 0
    assert not contains_sensitive(result_path.read_text(encoding="utf-8"))


def test_memory_ttl_drill_uses_injected_clock(tmp_path: Path) -> None:
    clock = Clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    store = MemoryStore(tmp_path / "memory.sqlite3", clock=clock)
    store.write_explicit("alice", "I am investigating the launch rollout")
    clock.advance_days(15)
    assert store.expire("alice", memory_type="project", max_age_days=14) == 1
    assert store.list_active("alice") == []
