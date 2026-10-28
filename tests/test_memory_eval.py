import json
from pathlib import Path

from asterline_assistant.config import Settings
from asterline_assistant.memory_eval import run_memory_eval


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
