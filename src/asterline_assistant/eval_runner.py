"""Resumable, deterministic evaluation with provider-error isolation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .corpus import load_corpus, validate_corpus_integrity
from .llm import ProviderFailure, make_provider
from .memory import MemoryStore
from .models import AnswerDraft, EvidencePack, HistoryMessage, ResearchResponse
from .research import build_evidence, retrieve, route, run_research
from .retrieval import HybridRetriever
from .tracing import clear_trace_context, set_trace_context

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
    return all(_matches(pattern, answer) for pattern in gold_facts) and all(not _matches(pattern, answer) or _match_has_marker(pattern, answer) for pattern in forbidden_facts)


def superseded_pass(answer: str, superseded_facts: list[str]) -> bool:
    return all(not _matches(pattern, answer) or _match_has_marker(pattern, answer) for pattern in superseded_facts)


def abstention_pass(answer: str, gold_facts: list[str], forbidden_facts: list[str]) -> bool:
    decline = bool(re.search(r"\b(cannot|can't|could not|couldn't|not published|not available|no information|no claims|unable to confirm|negotiated)\b", answer, re.IGNORECASE))
    return decline and fact_pass(answer, gold_facts, forbidden_facts)


def citation_pairs(answer: str) -> list[tuple[str, str]]:
    return re.findall(r"\[([^\s—]+)\s+—\s+([^\]]+)\]", answer)


def citation_ids(answer: str) -> list[str]:
    return [chunk_id for chunk_id, _ in citation_pairs(answer)]


def cache_key(row: dict[str, Any], provider: str, model: str, config: str, evidence: EvidencePack) -> str:
    evidence_hash = hashlib.sha256(json.dumps([item.chunk.model_dump(mode="json") for item in evidence.chunks], sort_keys=True).encode()).hexdigest()
    payload = {"provider": provider, "model": model, "config": config, "question": row["question"], "evidence_hash": evidence_hash}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _safe_error(error: Exception) -> tuple[str, str]:
    error_class = error.error_class if isinstance(error, ProviderFailure) else error.__class__.__name__
    message = re.sub(r"(?i)(api[_ -]?key|token|password|authorization)\s*[:=]?\s*[^\s,;]+", r"\1=<redacted>", str(error)).replace("\n", " ")[:200]
    return error_class, message or "evaluation case failed"


def _source_only_response(evidence: EvidencePack, provider: str, model: str, error_class: str, provider_error: bool) -> ResearchResponse:
    return ResearchResponse(answer="The language model is temporarily unavailable; here are the most relevant sources:", evidence=evidence.chunks, provider=provider, model=model, degraded=True, provider_error=provider_error, error_class=error_class, attempts=0)


def _case_result(row: dict[str, Any], response: ResearchResponse, status: str, elapsed_ms: float, calls: int) -> dict[str, Any]:
    retrieved = [item.chunk.chunk_id for item in response.evidence]
    gold = row.get("gold_chunks", [])
    cited = citation_pairs(response.answer)
    cited_ids = {chunk_id for chunk_id, _ in cited}
    gold_ids = set(gold)
    if status in {"provider_error", "eval_error"}:
        checks: dict[str, float | None] = {"retrieval_recall_at_5": float(bool(gold) and bool(gold_ids & set(retrieved))) if status == "provider_error" else None, "mrr": None, "fact_accuracy": None, "citation_validity": None, "citation_precision": None, "citation_recall": None, "conflict_handling": None, "abstention_accuracy": None}
    else:
        facts = abstention_pass(response.answer, row.get("gold_facts", []), row.get("forbidden_facts", [])) if row.get("expected") == "abstain" else fact_pass(response.answer, row.get("gold_facts", []), row.get("forbidden_facts", []))
        conflict = (facts and superseded_pass(response.answer, row.get("superseded_facts", []))) if row.get("superseded_facts") else None
        checks = {
            "retrieval_recall_at_5": float(bool(gold) and bool(gold_ids & set(retrieved))) if gold else None,
            "mrr": next((1 / (retrieved.index(chunk) + 1) for chunk in gold if chunk in retrieved), 0.0) if gold else None,
            "fact_accuracy": float(facts),
            "citation_validity": float(all(chunk_id in retrieved for chunk_id in cited_ids)),
            "citation_precision": len(cited_ids & gold_ids) / len(cited_ids) if gold and cited_ids else (0.0 if gold else None),
            "citation_recall": len(cited_ids & gold_ids) / len(gold_ids) if gold else None,
            "conflict_handling": float(conflict) if conflict is not None else None,
            "abstention_accuracy": float(response.abstained == (row.get("expected") == "abstain")),
        }
    return {"id": row["id"], "category": row["category"], "question": row["question"], "answer": response.answer, "status": status, "cited": [{"chunk_id": chunk_id, "title": title} for chunk_id, title in cited], "retrieved_not_cited": [chunk_id for chunk_id in retrieved if chunk_id not in cited_ids], "retrieved_chunk_ids": retrieved, "provider": response.provider, "model": response.model, "degraded": response.degraded, "provider_error": response.provider_error, "error_class": response.error_class, "checks": checks, "latency_ms": elapsed_ms, "llm_calls": calls}


def _summary(case_results: list[dict[str, Any]], rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = ["retrieval_recall_at_5", "mrr", "fact_accuracy", "citation_validity", "citation_precision", "citation_recall", "conflict_handling", "abstention_accuracy"]
    def metric_summary(results: list[dict[str, Any]], metric: str) -> dict[str, float | int]:
        values = [float(result["checks"][metric]) for result in results if result["checks"].get(metric) is not None]
        return {"value": _mean(values), "applicable": len(values), "total": len(results)}
    overall = {metric: metric_summary(case_results, metric) for metric in metrics}
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({result["category"] for result in case_results}):
        by_category[category] = {metric: metric_summary([result for result in case_results if result["category"] == category], metric) for metric in metrics}
    provider_errors = [{"id": result["id"], "category": result["category"], "provider": result["provider"], "error_class": result.get("error_class"), "reason": "language model providers unavailable"} for result in case_results if result["status"] == "provider_error"]
    eval_errors = [{"id": result["id"], "category": result["category"], "error_class": result.get("error_class"), "message": result.get("error_message", "evaluation case failed")} for result in case_results if result["status"] == "eval_error"]
    failures = [{"id": result["id"], "category": result["category"], "failed": [metric for metric, value in result["checks"].items() if value is not None and not value], "reason": "one or more deterministic checks failed"} for result in case_results if result["status"] == "ok" and any(value is not None and not value for value in result["checks"].values())]
    providers: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in case_results:
        providers[result["provider"]]["answers"] += int(result["status"] == "ok")
        providers[result["provider"]]["fallback_answers"] += int(result["degraded"] and result["status"] == "ok")
        providers[result["provider"]]["provider_errors"] += int(result["status"] == "provider_error")
    by_provider = {name: {**values, "fallback_fraction": values["fallback_answers"] / values["answers"] if values["answers"] else 0.0} for name, values in providers.items()}
    overall["provider_error_count"] = len(provider_errors)
    overall["eval_error_count"] = len(eval_errors)
    overall["llm_calls_per_question"] = _mean([result["llm_calls"] for result in case_results])
    served = sum(values["answers"] for values in providers.values())
    overall["fallback_fraction"] = {"value": sum(values["fallback_answers"] for values in providers.values()) / served if served else 0.0, "applicable": served, "total": len(case_results)}
    return overall, by_category | {"by_provider": by_provider}, provider_errors, failures + eval_errors


def run(dataset: Path, settings: Settings) -> Path:
    validate_corpus_integrity(settings.corpus_path)
    retriever = HybridRetriever(load_corpus(settings.corpus_path), settings.chroma_path, settings.fastembed_cache_path)
    provider = make_provider(settings.llm_provider, settings.gemini_model, settings.ollama_model)
    memory = MemoryStore(settings.sqlite_path)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache_path = Path("eval/results/cache/responses.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    output = Path("eval/results")
    output.mkdir(parents=True, exist_ok=True)
    output_path = output / f"{settings.config}_{provider.name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    case_results: list[dict[str, Any]] = []
    eval_run_id = uuid.uuid4().hex
    old_project = os.environ.get("LANGSMITH_PROJECT")
    os.environ["LANGSMITH_PROJECT"] = settings.eval_langsmith_project
    try:
        for row in rows:
            started = time.perf_counter()
            status = "ok"
            response: ResearchResponse
            evidence = EvidencePack(query=row["question"], chunks=[])
            key: str | None = None
            calls_before = getattr(provider, "calls", 0)
            error_message = None
            try:
                history = [HistoryMessage.model_validate(item) for item in row.get("history", [])]
                set_trace_context({"eval_run_id": eval_run_id, "case_id": row["id"], "category": row["category"], "config": settings.config, "provider": provider.name, "model": provider.model, "attempts": 0, "degraded": False})
                query = route(row["question"], history)
                evidence = build_evidence(retrieve(query, retriever))
                key = cache_key(row, provider.name, provider.model, settings.config, evidence)
                if key in cache:
                    cached = cache[key]
                    cached_response = cached.get("response", cached) if isinstance(cached, dict) else cached
                    response = ResearchResponse.model_validate(cached_response)
                    response.evidence = evidence.chunks
                    status = cached.get("status", "provider_error" if response.provider_error else "ok") if isinstance(cached, dict) else ("provider_error" if response.provider_error else "ok")
                    error_message = cached.get("error_message") if isinstance(cached, dict) else None
                else:
                    if settings.eval_delay_seconds > 0 and case_results:
                        time.sleep(settings.eval_delay_seconds)
                    response = run_research(row["question"], "eval", retriever, provider, memory, history=history, prepared_query=query, prepared_evidence=evidence)
                if response.provider_error:
                    status = "provider_error"
            except ProviderFailure as error:
                status = "provider_error"
                error_class, error_message = _safe_error(error)
                response = _source_only_response(evidence if "evidence" in locals() else EvidencePack(query=row["question"], chunks=[]), provider.name, provider.model, error_class, True)
            except Exception as error:
                status = "eval_error"
                error_class, error_message = _safe_error(error)
                response = _source_only_response(evidence if "evidence" in locals() else EvidencePack(query=row["question"], chunks=[]), provider.name, provider.model, error_class, False)
            finally:
                clear_trace_context()
            calls = max(getattr(provider, "calls", 0) - calls_before, 0)
            result = _case_result(row, response, status, (time.perf_counter() - started) * 1000, calls)
            if error_message:
                result["error_message"] = error_message
            case_results.append(result)
            if key is not None:
                cache[key] = {"response": response.model_dump(mode="json"), "status": status, "error_message": error_message}
            _atomic_write(cache_path, cache)
            overall, grouped, provider_errors, failures = _summary(case_results, rows[:len(case_results)])
            _atomic_write(output_path, {"status": "partial", "config": settings.config, "provider": provider.name, "model": provider.model, "cases": len(rows), "completed": len(case_results), "overall": overall, "by_category": {key: value for key, value in grouped.items() if key != "by_provider"}, "by_provider": grouped.get("by_provider", {}), "provider_errors": provider_errors, "failures": failures, "case_results": case_results})
            passed = status == "ok" and not any(value is not None and not value for value in result["checks"].values())
            print(f"{row['id']} {'PASS' if passed else status.upper() if status != 'ok' else 'FAIL'} {(time.perf_counter() - started):.2f}s")
        overall, grouped, provider_errors, failures = _summary(case_results, rows)
        report = {"status": "complete", "config": settings.config, "provider": provider.name, "model": provider.model, "cases": len(rows), "overall": overall, "by_category": {key: value for key, value in grouped.items() if key != "by_provider"}, "by_provider": grouped.get("by_provider", {}), "provider_errors": provider_errors, "failures": failures, "case_results": case_results}
        _atomic_write(output_path, report)
        print(json.dumps({"overall": overall, "by_provider": grouped.get("by_provider", {})}, indent=2))
        print("APPLICABLE_METRICS:")
        for metric in ("retrieval_recall_at_5", "mrr", "fact_accuracy", "citation_validity", "citation_precision", "citation_recall", "conflict_handling", "abstention_accuracy"):
            summary = overall[metric]
            print(f"{metric}: {summary['value']:.2f} ({summary['applicable']}/{summary['total']} applicable)")
        print("PROVIDER_ERRORS:")
        for item in provider_errors:
            print(json.dumps(item, sort_keys=True))
        print("FAILURES:")
        for item in failures:
            print(json.dumps(item, sort_keys=True))
        return output_path
    finally:
        if old_project is None:
            os.environ.pop("LANGSMITH_PROJECT", None)
        else:
            os.environ["LANGSMITH_PROJECT"] = old_project
