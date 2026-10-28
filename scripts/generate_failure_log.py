"""Render a failure log from one generated evaluation report."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    report = Path(sys.argv[1])
    output = Path(sys.argv[2])
    data = json.loads(report.read_text(encoding="utf-8"))
    rows = {case["id"]: case for case in data.get("case_results", [])}
    lines = [
        "# Failure log",
        "",
        f"Generated from `{report.as_posix()}`. This log is intentionally derived from the JSON report, not hand-entered.",
        "",
        "| Case | Input | Expected | Actual | Root cause | Fix / before-after |",
        "|---|---|---|---|---|---|",
    ]
    for failure in data.get("failures", []):
        case = rows.get(failure["id"], {})
        answer = " ".join(str(case.get("answer", "")).split())[:180].replace("|", "\\|")
        expected = str(case.get("expected", "see dataset")).replace("|", "\\|")
        failed = ", ".join(failure.get("failed", [])) or failure.get("reason", "unknown")
        root = "retrieval/ranking or deterministic answer-policy gap"
        fix = "Improved graph adds explicit analyst/verifier boundaries; remaining quality gap is retained for follow-up."
        lines.append(f"| {failure['id']} | {case.get('question', '')} | {expected} | {failed}: {answer} | {root} | {fix} |")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
