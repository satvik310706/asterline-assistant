"""Best-effort observability helpers; quality runs never depend on LangSmith."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def maybe_upload(dataset: Path | None = None, report: Path | None = None) -> str:
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return "LangSmith upload skipped: credentials absent."
    # Upload APIs vary by LangSmith version. Keep this hook explicit and
    # best-effort; callers get a log line rather than a failed evaluation.
    try:
        from langsmith import Client

        client = Client()
        if dataset is not None:
            client.upload_dataset(dataset_name=f"asterline-{dataset.stem}", data_type="llm")
        return "LangSmith upload attempted."
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
        payload = {"name": name, "user_id_hash": "synthetic-redacted", "config": "improved", "provider": "fake", "degraded": name == "genuine-failure", "outcome": outcome, "agent_tree": tree}
        (output / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (output / f"{name}.md").write_text(f"# {name}\n\n- user: `{user}` (synthetic)\n- outcome: {outcome}\n\n```text\n{tree}\n```\n", encoding="utf-8")
    (output / "exported_at.txt").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    return output
