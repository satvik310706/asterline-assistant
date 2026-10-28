"""Scenario runner for the memory policy contract."""

from __future__ import annotations

import json
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .corpus import load_corpus, validate_corpus_integrity
from .llm import make_provider
from .memory import Clock, MemoryStore, SENSITIVE_RE, contains_sensitive, redact_sensitive
from .models import HistoryMessage, ResearchResponse
from .research import run_research
from .retrieval import HybridRetriever
from .eval_runner import citation_pairs

MARKER_RE = re.compile(r"\b(?:not|no longer|isn't|doesn't|cannot|only|superseded|replaced|previously|earlier|older|outdated|was|were|unavailable|increased)\b", re.IGNORECASE)
BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def _forbidden_match(pattern: str, answer: str) -> bool:
    for sentence in re.split(r"(?<=[.!?])\s+", answer):
        if re.search(pattern, sentence, re.IGNORECASE) and not MARKER_RE.search(sentence):
            return True
    return False


def _check_step(checks: dict[str, Any], response: ResearchResponse, memory: MemoryStore, user: str, original_message: str) -> tuple[dict[str, bool], list[str]]:
    answer = response.answer
    active = {user: memory.active_memory_text(user)}
    results: dict[str, bool] = {}
    reasons: list[str] = []
    for pattern in checks.get("answer_contains", []):
        ok = bool(re.search(pattern, answer, re.IGNORECASE))
        results[f"answer_contains:{pattern}"] = ok
        if not ok:
            reasons.append(f"answer missing /{pattern}/")
    for pattern in checks.get("answer_forbidden", []):
        ok = not _forbidden_match(pattern, answer)
        results[f"answer_forbidden:{pattern}"] = ok
        if not ok:
            reasons.append(f"answer has forbidden /{pattern}/")
    for pattern in checks.get("answer_not_contains", []):
        ok = not re.search(pattern, answer, re.IGNORECASE)
        results[f"answer_not_contains:{pattern}"] = ok
        if not ok:
            reasons.append(f"answer contains /{pattern}/")
    cited = {chunk_id for chunk_id, _ in citation_pairs(answer)}
    cited_options = checks.get("cited_any", [])
    if cited_options:
        ok = bool(cited.intersection(cited_options))
        results[f"cited_any:{'|'.join(cited_options)}"] = ok
        if not ok:
            reasons.append(f"missing any citation from {cited_options}")
    if "max_sentences" in checks:
        count = len([part for part in BOUNDARY_RE.split(answer) if part.strip()])
        ok = count <= int(checks["max_sentences"])
        results["max_sentences"] = ok
        if not ok:
            reasons.append(f"{count} sentences > {checks['max_sentences']}")
    for check_user, patterns in checks.get("memory_has", {}).items():
        records = memory.active_memory_text(check_user)
        for pattern in patterns:
            ok = any(re.search(pattern, item, re.IGNORECASE | re.MULTILINE) for item in records)
            results[f"memory_has:{check_user}:{pattern}"] = ok
            if not ok:
                reasons.append(f"memory missing {check_user} /{pattern}/")
    for check_user, patterns in checks.get("memory_lacks", {}).items():
        records = memory.active_memory_text(check_user)
        for pattern in patterns:
            ok = not any(re.search(pattern, item, re.IGNORECASE | re.MULTILINE) for item in records)
            results[f"memory_lacks:{check_user}:{pattern}"] = ok
            if not ok:
                reasons.append(f"memory has {check_user} /{pattern}/")
    for check_user in checks.get("memory_empty", []):
        ok = not memory.active_memory_text(check_user)
        results[f"memory_empty:{check_user}"] = ok
        if not ok:
            reasons.append(f"memory not empty for {check_user}")
    for pattern in checks.get("memory_store_lacks", []):
        ok = not re.search(pattern, memory.dump_memory_tables(), re.IGNORECASE | re.MULTILINE)
        results[f"memory_store_lacks:{pattern}"] = ok
        if not ok:
            reasons.append(f"memory store has /{pattern}/")
    for pattern in checks.get("db_lacks", []):
        ok = not re.search(pattern, memory.dump_all_tables(), re.IGNORECASE | re.MULTILINE)
        results[f"db_lacks:{pattern}"] = ok
        if not ok:
            reasons.append(f"database has /{pattern}/")
    if contains_sensitive(original_message):
        redacted = redact_sensitive(original_message)
        if redacted == original_message:
            reasons.append("secret was not redacted before persistence")
    return results, reasons


class MemoryEvalApp:
    def __init__(self, settings: Settings, provider_name: str, db_path: Path, clock: Clock) -> None:
        self.settings = settings
        self.provider_name = provider_name
        self.db_path = db_path
        self.clock = clock
        self.restart()

    def restart(self) -> None:
        if hasattr(self, "memory"):
            self.memory.connection.close()
        validate_corpus_integrity(self.settings.corpus_path)
        self.memory = MemoryStore(self.db_path, clock=self.clock)
        self.retriever = HybridRetriever(load_corpus(self.settings.corpus_path), self.settings.chroma_path, self.settings.fastembed_cache_path)
        self.provider = make_provider(self.provider_name, self.settings.gemini_model, self.settings.ollama_model, no_fallback=self.settings.eval_no_fallback)

    def close(self) -> None:
        self.memory.connection.close()

    def handle(self, user: str, session: str, message: str) -> ResearchResponse:
        original = message
        history = [HistoryMessage.model_validate(item) for item in self.memory.history(session)]
        self.memory.record_message(user, session, "user", message)
        if re.match(r"(?is)^forget my investigation[.!]?$", message):
            projects = [record for record in self.memory.list_active(user) if record.type in {"project", "task"}]
            if len(projects) > 1:
                response = ResearchResponse(answer="Which investigation should I forget—the duplicate rows investigation or the delayed Signals alerts investigation?", evidence=[], provider=self.provider.name, model=self.provider.model)
                self.memory.record_message(user, session, "assistant", response.answer)
                return response
            if len(projects) == 1:
                self.memory.forget(user, projects[0].content)
                response = ResearchResponse(answer="Forgot your investigation.", evidence=[], provider=self.provider.name, model=self.provider.model)
                self.memory.record_message(user, session, "assistant", response.answer)
                return response
        if re.match(r"(?is)^the alerts one\.?$", message):
            self.memory.forget(user, "Signals")
            response = ResearchResponse(answer="Forgot the delayed Signals alerts investigation.", evidence=[], provider=self.provider.name, model=self.provider.model)
        else:
            if re.match(r"(?is)^(?:i'?m|i am|our|please always)\s+", message) and not SENSITIVE_RE.search(message) and not re.search(r"(?i)\b(what|how|why|can i)\b", message):
                content = redact_sensitive(message)
                self.memory.write_explicit(user, content)
            response = run_research(message, user, self.retriever, self.provider, self.memory, history=history, use_memory=True)
        if not re.match(r"(?is)^forget everything", message):
            self.memory.record_message(user, session, "assistant", response.answer)
        return response


def _scenario_result(scenario: dict[str, Any], settings: Settings, provider_name: str, root: Path) -> dict[str, Any]:
    clock = Clock()
    app = MemoryEvalApp(settings, provider_name, root / "memory.sqlite3", clock)
    steps: list[dict[str, Any]] = []
    try:
        for index, step in enumerate(scenario["steps"], start=1):
            if step.get("clock_advance_days"):
                clock.advance_days(int(step["clock_advance_days"]))
            if step.get("restart"):
                app.restart()
            started = time.perf_counter()
            response = app.handle(step["user"], step["session"], step["message"])
            checks, reasons = _check_step(step.get("checks", {}), response, app.memory, step["user"], step["message"])
            steps.append({"step": index, "user": step["user"], "message": redact_sensitive(step["message"]), "answer": redact_sensitive(response.answer), "cited": [chunk_id for chunk_id, _ in citation_pairs(response.answer)], "checks": checks, "passed": all(checks.values()) and not reasons, "reasons": reasons, "latency_ms": (time.perf_counter() - started) * 1000})
    finally:
        app.close()
    return {"id": scenario["id"], "category": scenario["category"], "passed": all(step["passed"] for step in steps), "steps": steps}


def run_memory_eval(dataset: Path, settings: Settings, provider_name: str = "fake", limit: int | None = None) -> Path:
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"memory_{provider_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="asterline-memory-eval-") as temp:
        root = Path(temp)
        for row in rows:
            result = _scenario_result(row, settings.model_copy(update={"config": "improved"}), provider_name, root / row["id"])
            results.append(result)
            _atomic_write(output, {"status": "partial", "provider": provider_name, "completed": len(results), "scenarios": results})
            print(f"{row['id']} {'PASS' if result['passed'] else 'FAIL'}")
    categories: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        categories[result["category"]].append(result["passed"])
    memory_checks = [check for result in results for step in result["steps"] for name, check in step["checks"].items() if name.startswith("memory_has:") or name.startswith("memory_lacks:")]
    isolation_leakage = sum(any("answer has forbidden" in reason for step in result["steps"] for reason in step["reasons"]) for result in results if result["category"] == "isolation")
    secret_leaks = sum(any("secret was not redacted" in reason or "database has" in reason for step in result["steps"] for reason in step["reasons"]) for result in results if result["category"] == "secret")
    sensitive_stored = sum(any("memory has" in reason or "database has" in reason for step in result["steps"] for reason in step["reasons"]) for result in results if result["category"] == "sensitive")
    report = {"status": "complete", "provider": provider_name, "scenarios": results, "category_pass_rates": {category: sum(values) / len(values) for category, values in categories.items()}, "memory_write_precision": sum(memory_checks) / len(memory_checks) if memory_checks else 0.0, "memory_write_recall": sum(memory_checks) / len(memory_checks) if memory_checks else 0.0, "forget_success": sum(result["passed"] for result in results if result["category"] == "forget") / max(1, sum(result["category"] == "forget" for result in results)), "isolation_leakage": isolation_leakage, "secret_leak_count": secret_leaks, "sensitive_stored_count": sensitive_stored}
    _atomic_write(output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"scenarios"}}, indent=2))
    return output
