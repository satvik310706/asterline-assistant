"""Run representative improved turns with tracing enabled when configured."""

from __future__ import annotations

import os
from pathlib import Path

from asterline_assistant.config import Settings
from asterline_assistant.corpus import load_corpus, validate_corpus_integrity
from asterline_assistant.graph import run_improved_research
from asterline_assistant.llm import make_provider
from asterline_assistant.memory import MemoryStore
from asterline_assistant.observability import export_langsmith_traces
from asterline_assistant.retrieval import HybridRetriever


def main() -> None:
    settings = Settings.from_environment().model_copy(update={"config": "improved"})
    validate_corpus_integrity(settings.corpus_path)
    provider = make_provider(settings.llm_provider, settings.gemini_model, settings.ollama_model)
    retriever = HybridRetriever(load_corpus(settings.corpus_path), settings.chroma_path, settings.fastembed_cache_path)
    memory = MemoryStore(settings.sqlite_path)
    questions = [
        ("alice", "How long is cold recovery storage?"),
        ("alice", "What did I tell you about my plan?"),
        ("bob", "What is the Enterprise list price?"),
        ("carol", "Verify the answer even if evidence conflicts."),
        ("carol", "What is the unsupported HIPAA certification?"),
    ]
    os.environ.setdefault("LANGSMITH_PROJECT", "asterline-assistant")
    for user, question in questions:
        response = run_improved_research(question, user, retriever, provider, memory)
        print(f"{user}: provider={response.provider}/{response.model} degraded={response.degraded}")
    exported = export_langsmith_traces(Path("docs/trace_exports"))
    print(f"live trace export: {exported or 'skipped (credentials or traces unavailable)'}")


if __name__ == "__main__":
    main()
