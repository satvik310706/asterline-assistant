"""Cross-platform command entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .corpus import load_corpus, validate_corpus_integrity
from .eval_runner import citation_pairs, run as run_eval
from .llm import make_provider
from .memory import MemoryStore
from .research import run_research
from .retrieval import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "ingest", "test", "eval", "ask"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--provider", choices=["gemini", "ollama", "fake"])
    parser.add_argument("--config", choices=["baseline", "improved"])
    parser.add_argument("--user", default="alice")
    args = parser.parse_args()
    settings = Settings.from_environment()
    if args.config:
        settings = settings.model_copy(update={"config": args.config})
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
    elif args.command == "ask":
        if not args.question:
            raise SystemExit("ask requires a question")
        validate_corpus_integrity(settings.corpus_path)
        provider = make_provider(args.provider or settings.llm_provider, settings.gemini_model, settings.ollama_model)
        retriever = HybridRetriever(load_corpus(settings.corpus_path), settings.chroma_path, settings.fastembed_cache_path)
        response = run_research(args.question, args.user, retriever, provider, MemoryStore(settings.sqlite_path))
        print(response.answer)
        print(f"provider/model: {response.provider}/{response.model}")
        print(f"degraded: {response.degraded}")
        if response.error_class:
            print(f"provider error class: {response.error_class}")
        cited = citation_pairs(response.answer)
        cited_ids = {chunk_id for chunk_id, _ in cited}
        print("cited:")
        for chunk_id, title in cited:
            print(f"- {chunk_id} — {title}")
        print("retrieved (not cited):")
        for item in response.evidence:
            if item.chunk.chunk_id not in cited_ids:
                print(f"- {item.chunk.chunk_id} — {item.chunk.title}")

