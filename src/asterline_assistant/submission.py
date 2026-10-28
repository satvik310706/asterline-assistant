"""Build the single-file take-home submission from generated project artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_simple_pdf(path: Path, lines: list[str]) -> None:
    pages = [lines[index:index + 46] for index in range(0, len(lines), 46)] or [["Asterline Assistant"]]
    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    page_ids = [3 + index * 2 for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    for index, page_lines in enumerate(pages):
        page_id = page_ids[index]
        stream_id = page_id + 1
        content = ["BT", "/F1 9 Tf", "45 760 Td"]
        for line in page_lines:
            content.append(f"({_pdf_escape(line[:110])}) Tj")
            content.append("0 -15 Td")
        content.append("ET")
        stream = "\n".join(content).encode("latin-1", "replace")
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {3 + len(pages) * 2} 0 R >> >> /Contents {stream_id} 0 R >>".encode())
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    font_id = 3 + len(pages) * 2
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(output)


def build_submission(root: Path = Path(".")) -> tuple[Path, Path]:
    final = root / "eval" / "final"
    reports = []
    for path in sorted(final.glob("*.json")):
        reports.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
    eval_text = (root / "docs" / "EVAL.md").read_text(encoding="utf-8") if (root / "docs" / "EVAL.md").exists() else "No generated evaluation report."
    design = (root / "docs" / "DESIGN.md").read_text(encoding="utf-8") if (root / "docs" / "DESIGN.md").exists() else ""
    ai_use = (root / "docs" / "AI_USE.md").read_text(encoding="utf-8") if (root / "docs" / "AI_USE.md").exists() else ""
    lines = [
        "# Asterline Assistant — Take-home submission",
        "",
        "Repository: [PASTE_REPOSITORY_URL_HERE]",
        "LangSmith traces: [PASTE_SHARED_TRACE_LINKS_HERE]",
        "",
        "## Executive summary",
        "A research assistant that answers only from an immutable corpus, cites evidence,",
        "handles supersession and abstention, and remembers safe user context across restarts.",
        "The baseline is frozen; the improved configuration adds seven typed agent stages.",
        "",
        "## Reproduction",
        "uv sync",
        "uv run pytest -q",
        "uv run asterline ask \"What is the current Launch ingestion rate?\" --provider fake",
        "uv run asterline run",
        "",
        "## Generated evaluation artifacts",
    ]
    for name, data in reports:
        lines.append(f"{name}: provider={data.get('provider')} config={data.get('config')} cases={data.get('cases')}")
        for metric, value in data.get("overall", {}).items():
            if isinstance(value, dict) and "value" in value:
                lines.append(f"  {metric}: {value['value']} ({value.get('applicable', 0)}/{value.get('total', 0)} applicable)")
    lines += [
        "",
        "## Evidence and memory guarantees",
        "Citations are parsed and validated against retrieved chunks. Memory is labelled",
        "unverified context and never counts as company evidence. Secrets and sensitive",
        "personal data are redacted or rejected before storage and tracing. SQLite is the",
        "source of truth; hard delete removes memory rows, vectors, sessions, and messages.",
        "",
        "## Provider resilience smoke",
        "A real Gemini request returned HTTP 429 (quota exhausted); the app recorded the",
        "sanitized primary error and successfully served the answer through the locally",
        "installed Ollama llama3.2:3b model with a citation validated against retrieved evidence.",
        "This is a resilience smoke test, not a Gemini answer-quality evaluation.",
        "",
        "## Limitations",
        "FakeLLM reports verify plumbing, not provider quality. Gemini quality runs and",
        "remote LangSmith links require the author's credentials and quota. Deleted data",
        "may remain in already-shipped remote traces. Regex matching is a rough metric.",
        "",
        "## AI use disclosure",
    ]
    lines.extend(ai_use.splitlines())
    lines += ["", "## Design excerpt"] + design.splitlines()[:35]
    markdown = "\n".join(lines) + "\n"
    md_path = root / "SUBMISSION.md"
    pdf_path = root / "SUBMISSION.pdf"
    md_path.write_text(markdown, encoding="utf-8")
    pdf_lines = [re.sub(r"[`*_]", "", line) for line in lines]
    _write_simple_pdf(pdf_path, pdf_lines)
    return md_path, pdf_path
