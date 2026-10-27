"""Cross-platform command entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .corpus import load_corpus, validate_corpus_integrity
from .eval_runner import run as run_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "ingest", "test", "eval"])
    args = parser.parse_args()
    settings = Settings.from_environment()
    if args.command == "ingest":
        validate_corpus_integrity(settings.corpus_path)
        print(f"Loaded {len(load_corpus(settings.corpus_path))} corpus chunks.")
    elif args.command == "eval":
        dataset = Path("eval/datasets/research.jsonl")
        if not dataset.exists():
            raise SystemExit("Evaluation dataset not found: eval/datasets/research.jsonl")
        print(run_eval(dataset, settings))
    elif args.command == "test":
        import pytest
        raise SystemExit(pytest.main(["tests"]))
    elif args.command == "run":
        validate_corpus_integrity(settings.corpus_path)
        print("Asterline baseline is ready. Streamlit UI is deferred to a later phase.")

