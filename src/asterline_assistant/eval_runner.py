"""Deterministic evaluation runner for the baseline and future configurations."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .corpus import load_corpus, validate_corpus_integrity
from .llm import make_provider
from .memory import MemoryStore
from .models import HistoryMessage, ResearchResponse
from .research import run_research
from .retrieval import HybridRetriever

MARKERS = re.compile(r"\b(?:not|no longer|isn't|doesn't|cannot|only|superseded|replaced|previously|earlier|older|outdated|was|were|unavailable|increased)\b", re.IGNORECASE)


def _matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error:
        return pattern.casefold() in text.casefold()


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _match_has_marker(pattern: str, text: str) -> bool:
    return any(_matches(pattern, sentence) and MARKERS.search(sentence) for sentence in _sentences(text))


def fact_pass(answer: str, gold_facts: list[str], forbidden_facts: list[str]) -> bool:
    return all(_matches(pattern, answer) for pattern in gold_facts) and all(
        not _matches(pattern, answer) or _match_has_marker(pattern, answer) for pattern in forbidden_facts
    )


def superseded_pass(answer: str, superseded_facts: list[str]) -> bool:
    return all(not _matches(pattern, answer) or _match_has_marker(pattern, answer) for pattern in superseded_facts)


def abstention_pass(answer: str, gold_facts: list[str], forbidden_facts: list[str]) -> bool:
    decline = bool(re.search(r"\b(cannot|can't|could not|couldn't|not published|not available|no information|no claims|unable to confirm|negotiated)\b", answer, re.IGNORECASE))
    return decline and fact_pass(answer, gold_facts, forbidden_facts)


def citation_ids(answer: str) -> list[str]:
    return re.findall(r"\[([^\s—]+)\s+—\s+[^\]]+\]", answer)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run(dataset: Path, settings: Settings) -> Path:
    validate_corpus_integrity(settings.corpus_path)
    retriever = HybridRetriever(load_corpus(settings.corpus_path), settings.chroma_path)
    provider = make_provider(settings.llm_provider, settings.gemini_model, settings.ollama_model)
    memory = MemoryStore(settings.sqlite_path)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache_path = Path("eval/results/cache/responses.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    case_results: list[dict[str, Any]] = []
    category_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        started = time.perf_counter()
        key = hashlib.sha256(json.dumps({"row": row, "provider": provider.name, "model": provider.model}, sort_keys=True).encode()).hexdigest()
        before_calls = getattr(provider, "calls", 0)
        if key in cache:
            response = ResearchResponse.model_validate(cache[key])
        else:
            history = [HistoryMessage.model_validate(item) for item in row.get("history", [])]
            response = run_research(row["question"], "eval", retriever, provider, memory, history=history)
            cache[key] = response.model_dump(mode="json")
        calls = max(getattr(provider, "calls", 0) - before_calls, 0)
        retrieved = [item.chunk.chunk_id for item in response.evidence]
        gold = row.get("gold_chunks", [])
        reciprocal_rank = next((1 / (retrieved.index(chunk) + 1) for chunk in gold if chunk in retrieved), 0.0)
        recall = float(not gold or any(chunk in retrieved for chunk in gold))
        facts = abstention_pass(response.answer, row.get("gold_facts", []), row.get("forbidden_facts", [])) if row.get("expected") == "abstain" else fact_pass(response.answer, row.get("gold_facts", []), row.get("forbidden_facts", []))
        superseded = superseded_pass(response.answer, row.get("superseded_facts", []))
        conflict = facts and superseded if row.get("superseded_facts") else True
        citations = citation_ids(response.answer)
        citation_valid = all(chunk_id in retrieved for chunk_id in citations)
        abstention = response.abstained == (row.get("expected") == "abstain")
        elapsed_ms = (time.perf_counter() - started) * 1000
        checks = {"retrieval_recall_at_5": recall, "fact_accuracy": float(facts), "citation_validity": float(citation_valid), "conflict_handling": float(conflict), "abstention_accuracy": float(abstention)}
        result = {"id": row["id"], "category": row["category"], "question": row["question"], "answer": response.answer, "cited_ids": citations, "retrieved_chunk_ids": retrieved, "checks": checks, "latency_ms": elapsed_ms, "llm_calls": calls}
        case_results.append(result)
        for metric, value in checks.items():
            if metric != "retrieval_recall_at_5" or gold:
                category_values[row["category"]][metric].append(value)
        if gold:
            category_values[row["category"]]["mrr"].append(reciprocal_rank)

    overall = {metric: _mean([result["checks"][metric] for result in case_results]) for metric in ["fact_accuracy", "citation_validity", "conflict_handling", "abstention_accuracy"]}
    retrieval_cases = [result for result, row in zip(case_results, rows) if row.get("gold_chunks")]
    overall["recall_at_5"] = _mean([result["checks"]["retrieval_recall_at_5"] for result in retrieval_cases])
    overall["mrr"] = _mean([next((1 / (result["retrieved_chunk_ids"].index(chunk) + 1) for chunk in row.get("gold_chunks", []) if chunk in result["retrieved_chunk_ids"]), 0.0) for result, row in zip(case_results, rows) if row.get("gold_chunks")])
    overall["mean_latency_ms"] = _mean([result["latency_ms"] for result in case_results])
    overall["llm_calls_per_question"] = _mean([result["llm_calls"] for result in case_results])
    category_summary = {category: {metric: _mean(values) for metric, values in metrics.items()} for category, metrics in category_values.items()}
    failures = [{"id": result["id"], "category": result["category"], "failed": [metric for metric, passed in result["checks"].items() if not passed], "reason": "one or more deterministic checks failed"} for result in case_results if not all(result["checks"].values())]
    output = Path("eval/results")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{settings.config}_{provider.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    report = {"config": settings.config, "provider": provider.name, "model": provider.model, "cases": len(rows), "overall": overall, "by_category": category_summary, "failures": failures, "case_results": case_results}
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    print(json.dumps({"config": settings.config, "provider": provider.name, "model": provider.model, "cases": len(rows), "overall": overall, "by_category": category_summary}, indent=2))
    print("FAILURES:")
    for failure in failures:
        print(json.dumps(failure, sort_keys=True))
    return path
