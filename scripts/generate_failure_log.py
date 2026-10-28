"""Render a reproducible before/after failure log from evaluation JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


METRICS = (
    "retrieval_recall_at_5",
    "mrr",
    "fact_accuracy",
    "citation_validity",
    "citation_precision",
    "citation_recall",
    "conflict_handling",
    "abstention_accuracy",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cases(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in report.get("case_results", [])}


def _summary(result: dict[str, Any] | None) -> str:
    if result is None:
        return "not present in this run"
    checks = result.get("checks", {})
    failed = [name for name, value in checks.items() if value is False]
    status = str(result.get("status", "unknown"))
    error = result.get("error_class")
    details = f"{status}" + (f" ({error})" if error else "")
    if failed:
        details += "; failed=" + ", ".join(failed)
    answer = " ".join(str(result.get("answer", "")).split())[:180]
    if answer:
        details += "; answer=" + answer
    return details.replace("|", "\\|")


def _metric(report: dict[str, Any], name: str) -> str:
    value = report.get("overall", {}).get(name)
    if isinstance(value, dict):
        return f"{value.get('value', 0):.4f} ({value.get('applicable', 0)}/{value.get('total', 0)} applicable)"
    if isinstance(value, (int, float)):
        return f"{value}"
    return "not reported"


def _root_cause(before: dict[str, Any] | None, after: dict[str, Any] | None) -> tuple[str, str]:
    if before and before.get("status") == "provider_error":
        error_class = before.get("error_class") or "provider failure"
        if error_class == "TimeoutError" and after and after.get("status") != "provider_error":
            return (
                "The local Ollama request exceeded the previous 12-second socket timeout; the same case later completed.",
                "Raised the local request timeout to 45 seconds. Compare provider-error counts and latency in the run summaries.",
            )
        if error_class == "TimeoutError":
            return ("The local Ollama request exceeded its socket timeout.", "The 45-second timeout did not resolve this case; retain it as an unresolved provider failure.")
        return (f"The provider failed with {error_class}.", "No quality fix is claimed; inspect the sanitized provider diagnostics.")

    before_checks = (before or {}).get("checks", {})
    after_checks = (after or {}).get("checks", {})
    before_failed = [name for name, value in before_checks.items() if value is False]
    after_failed = [name for name, value in after_checks.items() if value is False]
    category = (after or before or {}).get("category")
    if category == "arithmetic" and "fact_accuracy" in before_failed and "fact_accuracy" not in after_failed:
        return (
            "The earlier answer failed exact arithmetic while the supplied source facts were available.",
            "The improved path now uses a deterministic evidence-backed calculator; this case's after-run passed fact matching.",
        )

    checks = (after or before or {}).get("checks", {})
    failed = [name for name, value in checks.items() if value is False]
    if "retrieval_recall_at_5" in failed or "mrr" in failed:
        return ("One or more gold chunks were not retrieved or ranked well.", "No retrieval fix is claimed in this run; use the retrieved IDs and gold chunks for the next change.")
    if "citation_validity" in failed:
        return ("The answer cited a chunk that was not retrieved.", "Keep citation validation strict; the case remains a quality failure.")
    if "conflict_handling" in failed:
        return ("The response did not both state the current value and correctly flag the superseded value.", "Keep conflict handling unresolved unless the measured after-run passes this check.")
    if "abstention_accuracy" in failed:
        return ("The response did not match the dataset's expected abstention behavior.", "Keep abstention handling unresolved unless the measured after-run passes this check.")
    if "fact_accuracy" in failed:
        return ("The answer did not match every deterministic gold-fact pattern.", "No factual-quality fix is claimed; inspect the cited evidence and answer before changing the policy.")
    if failed:
        return ("One or more deterministic checks failed: " + ", ".join(failed) + ".", "No fix is claimed for this case.")
    return ("No deterministic failure is recorded in this case.", "No change claimed.")


def render(before_path: Path, after_path: Path, dataset_path: Path) -> str:
    before_report = _load(before_path)
    after_report = _load(after_path)
    rows = {
        row["id"]: row
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    before_cases = _cases(before_report)
    after_cases = _cases(after_report)
    failures: list[str] = []
    for case_id in sorted(before_cases.keys() | after_cases.keys()):
        before = before_cases.get(case_id)
        after = after_cases.get(case_id)
        before_failed = before and (before.get("status") == "provider_error" or any(value is False for value in before.get("checks", {}).values()))
        after_failed = after and (after.get("status") == "provider_error" or any(value is False for value in after.get("checks", {}).values()))
        if not (before_failed or after_failed):
            continue
        source = after or before or {}
        question = str(source.get("question", rows.get(case_id, {}).get("question", ""))).replace("|", "\\|")
        expected = str(rows.get(case_id, {}).get("expected", "unknown")).replace("|", "\\|")
        cause, fix = _root_cause(before, after)
        failures.append(
            f"| {case_id} | {question} | {expected} | {_summary(before)} | {_summary(after)} | {cause} | {fix} |"
        )

    lines = [
        "# Failure log",
        "",
        f"Generated from `{before_path.as_posix()}` and `{after_path.as_posix()}`. Metrics and case rows are read from JSON; no result is hand-entered.",
        "",
        f"Paired provider/model/config: `{before_report.get('provider')}/{before_report.get('model')}/{before_report.get('config')}` → `{after_report.get('provider')}/{after_report.get('model')}/{after_report.get('config')}`.",
        "",
        "| Metric | Before | After |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(f"| {metric} | {_metric(before_report, metric)} | {_metric(after_report, metric)} |")
    lines += [
        f"| provider_error_count | {_metric(before_report, 'provider_error_count')} | {_metric(after_report, 'provider_error_count')} |",
        "",
        "## Failed or provider-error cases",
        "",
        "| Case | Input | Expected | Before | After | Root cause | Fix / remaining status |",
        "|---|---|---|---|---|---|---|",
        *(failures or ["| — | No failures recorded | — | — | — | — | — |"]),
        "",
        "This log describes measured behavior; a provider-error reduction is not an answer-quality improvement. FakeLLM metrics are not mixed into this comparison.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: generate_failure_log.py BEFORE.json AFTER.json research.jsonl")
    before_path, after_path, dataset_path = map(Path, sys.argv[1:])
    output = Path("docs/FAILURE_LOG.md")
    output.write_text(render(before_path, after_path, dataset_path), encoding="utf-8")


if __name__ == "__main__":
    main()
