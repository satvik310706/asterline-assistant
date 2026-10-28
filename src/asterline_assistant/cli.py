"""Cross-platform command entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .corpus import load_corpus, validate_corpus_integrity
from .eval_runner import citation_pairs, run as run_eval
from .memory_eval import run_memory_eval
from .llm import ProviderFailure, format_failure, make_provider
from .memory import MemoryStore
from .research import run_research
from .graph import run_improved_research
from .retrieval import HybridRetriever
from .observability import export_langsmith_traces, export_representative_traces, maybe_upload
from .reporting import publish_results, render_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "ingest", "test", "eval", "eval-memory", "ask", "export-traces", "publish-results", "report"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--provider", choices=["gemini", "ollama", "fake"])
    parser.add_argument("--config", choices=["baseline", "improved"])
    parser.add_argument("--user", default="alice")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_environment()
    if args.config:
        settings = settings.model_copy(update={"config": args.config})
    if args.command == "ingest":
        validate_corpus_integrity(settings.corpus_path)
        print(f"Loaded {len(load_corpus(settings.corpus_path))} corpus chunks.")
    elif args.command == "export-traces":
        print(export_representative_traces(Path("docs/trace_exports")))
        print(export_langsmith_traces(Path("docs/trace_exports")) or "LangSmith live trace export skipped.")
        print(maybe_upload(Path("eval/datasets/research.jsonl")))
    elif args.command == "publish-results":
        for report in publish_results():
            print(report)
    elif args.command == "report":
        print(render_eval())
    elif args.command == "eval":
        dataset = Path(args.dataset) if args.dataset else Path("eval/datasets/research.jsonl")
        if not dataset.exists():
            raise SystemExit("Evaluation dataset not found: eval/datasets/research.jsonl")
        if args.no_fallback:
            settings = settings.model_copy(update={"eval_no_fallback": True})
        print(run_eval(dataset, settings, limit=args.limit, skip_preflight=args.skip_preflight))
    elif args.command == "eval-memory":
        dataset = Path(args.dataset) if args.dataset else Path("eval/datasets/memory.jsonl")
        if not dataset.exists():
            raise SystemExit("Memory evaluation dataset not found: eval/datasets/memory.jsonl")
        if args.no_fallback:
            settings = settings.model_copy(update={"eval_no_fallback": True})
        print(run_memory_eval(dataset, settings, provider_name=args.provider or settings.llm_provider, limit=args.limit))
    elif args.command == "test":
        import pytest
        raise SystemExit(pytest.main(["tests"]))
    elif args.command == "run":
        validate_corpus_integrity(settings.corpus_path)
        import subprocess
        raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"]))
    elif args.command == "ask":
        if not args.question:
            raise SystemExit("ask requires a question")
        validate_corpus_integrity(settings.corpus_path)
        provider = make_provider(args.provider or settings.llm_provider, settings.gemini_model, settings.ollama_model)
        retriever = HybridRetriever(load_corpus(settings.corpus_path), settings.chroma_path, settings.fastembed_cache_path)
        research_fn = run_improved_research if settings.config == "improved" else run_research
        try:
            response = research_fn(args.question, args.user, retriever, provider, MemoryStore(settings.sqlite_path))
        except ProviderFailure as error:
            print("The language model is temporarily unavailable; here are the most relevant sources:")
            print(format_failure(provider.name, provider.model, error, "primary"))
            raise SystemExit(0)
        print(response.answer)
        print(f"provider/model: {response.provider}/{response.model}")
        print(f"degraded: {response.degraded}")
        if response.error_class:
            print(f"provider error class: {response.error_class}")
        if response.primary_error:
            print(response.primary_error)
        if response.fallback_error:
            print(response.fallback_error)
        cited = citation_pairs(response.answer)
        cited_ids = {chunk_id for chunk_id, _ in cited}
        print("cited:")
        for chunk_id, title in cited:
            print(f"- {chunk_id} — {title}")
        print("retrieved (not cited):")
        for item in response.evidence:
            if item.chunk.chunk_id not in cited_ids:
                print(f"- {item.chunk.chunk_id} — {item.chunk.title}")

