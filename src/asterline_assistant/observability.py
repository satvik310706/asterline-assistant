"""Best-effort observability helpers; quality runs never depend on LangSmith."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .memory import redact_sensitive


def _redact_payload(value: Any) -> Any:
    """Redact every string leaf before sending synthetic examples to LangSmith."""
    if isinstance(value, str):
        return redact_sensitive(value)
    if isinstance(value, dict):
        return {key: _redact_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


def maybe_upload(dataset: Path | None = None, report: Path | None = None) -> str:
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return "LangSmith upload skipped: credentials absent."
    try:
        from langsmith import Client

        dataset_name = f"asterline-{dataset.stem}" if dataset is not None else "asterline-eval"
        dataset_rows: list[dict] = []
        if dataset is not None:
            for line in dataset.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                dataset_rows.append(json.loads(line))
        report_data = {}
        if report is not None and report.exists():
            report_data = json.loads(report.read_text(encoding="utf-8"))
        report_cases = report_data.get("case_results", [])
        cases_by_id = {str(case.get("id")): case for case in report_cases if isinstance(case, dict) and case.get("id") is not None}
        examples = []
        uploaded_ids: set[str] = set()
        for row in dataset_rows:
            case_id = str(row.get("id", ""))
            case = cases_by_id.get(case_id, {})
            outputs = {
                "expected": row.get("expected"),
                "gold_chunks": row.get("gold_chunks", []),
                "gold_facts": row.get("gold_facts", []),
            }
            if case:
                outputs.update({"answer": case.get("answer", ""), "status": case.get("status"), "cited": case.get("cited", [])})
                uploaded_ids.add(case_id)
            examples.append(_redact_payload({
                "inputs": {"question": row.get("question", ""), "history": row.get("history", [])},
                "outputs": outputs,
                "metadata": {
                    "case_id": case_id,
                    "category": row.get("category"),
                    "synthetic": True,
                    "config": report_data.get("config"),
                    "provider": report_data.get("provider"),
                },
            }))
        for case_id, case in cases_by_id.items():
            if case_id in uploaded_ids:
                continue
            examples.append(_redact_payload({
                "inputs": {"question": case.get("question", "")},
                "outputs": {"answer": case.get("answer", ""), "status": case.get("status"), "cited": case.get("cited", [])},
                "metadata": {"case_id": case_id, "synthetic": True, "config": report_data.get("config"), "provider": report_data.get("provider")},
            }))
        if examples:
            client = Client()
            created = client.create_dataset(dataset_name=dataset_name, description="Asterline synthetic evaluation dataset and results")
            client.create_examples(dataset_id=created.id, examples=examples[:1000], max_concurrency=3)
            return f"LangSmith upload attempted for {dataset_name} ({min(len(examples), 1000)} examples)."
        return f"LangSmith upload skipped: no examples found for {dataset_name}."
    except Exception as error:
        return f"LangSmith upload skipped after non-fatal error: {error.__class__.__name__}."


def export_representative_traces(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    examples = [
        ("conflict-answer", "alice", "router → memory_reader → retriever → evidence_analyst → answer_writer → verifier → memory_writer", "Current evidence stated; superseded value flagged."),
        ("memory-followup-restart", "alice", "router → memory_reader → retriever → evidence_analyst → answer_writer → verifier → memory_writer", "UNVERIFIED USER CONTEXT used for disambiguation; not cited as evidence."),
        ("abstention", "bob", "router → retriever → evidence_analyst → answer_writer → verifier", "Information not published in the corpus."),
        ("verifier-repair", "carol", "router → retriever → evidence_analyst → answer_writer → verifier → answer_writer → verifier", "Invalid citation repaired once."),
        ("genuine-failure", "carol", "router → retriever → answer_writer", "Provider unavailable; sources returned in degraded mode."),
    ]
    for name, user, tree, outcome in examples:
        payload = {"name": name, "synthetic": True, "user_id_hash": "synthetic-redacted", "config": "improved", "provider": "fake", "degraded": name == "genuine-failure", "outcome": outcome, "agent_tree": tree}
        (output / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (output / f"{name}.md").write_text(f"# {name}\n\n- user: `{user}` (synthetic)\n- outcome: {outcome}\n\n```text\n{tree}\n```\n", encoding="utf-8")
    (output / "exported_at.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return output


def export_langsmith_traces(output: Path, project: str = "asterline-assistant") -> Path | None:
    """Export a small redacted sample when LangSmith is configured.

    The local synthetic export remains the fallback so documentation never
    implies that a remote trace was captured when the account was unavailable.
    """
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return None
    try:
        from langsmith import Client

        output.mkdir(parents=True, exist_ok=True)
        runs = list(Client().list_runs(project_name=project, limit=25))
        for index, run in enumerate(runs[:5], start=1):
            payload = {
                "synthetic": False,
                "id": str(getattr(run, "id", "")),
                "name": getattr(run, "name", ""),
                "run_type": getattr(run, "run_type", ""),
                "error": getattr(run, "error", None),
                "inputs": redact_sensitive(json.dumps(getattr(run, "inputs", {}), default=str)),
                "outputs": redact_sensitive(json.dumps(getattr(run, "outputs", {}), default=str)),
                "extra": redact_sensitive(json.dumps(getattr(run, "extra", {}), default=str)),
            }
            (output / f"live-{index:02d}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return output
    except Exception:
        return None
