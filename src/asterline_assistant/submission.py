"""Build the single-file take-home submission from generated project artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path

NAME = "Satvik Pedamallu"
NIAT_ID = "N24H01A0514"
GITHUB_URL = "https://github.com/satvik310706/asterline-assistant"
LANGSMITH_URL = "https://smith.langchain.com/o/34ea5156-f488-4661-b611-d4fa3993ddf4/a/asterline-assistant/projects/p/e99ab585-0198-416a-b09f-471470f3df09"


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


def _generate_rich_pdf(output_pdf: Path, root: Path) -> bool:
    """Render publication-grade PDF using Playwright Chromium if available."""
    try:
        from playwright.sync_api import sync_playwright

        html_script = root / "scripts" / "generate_submission_pdf.py"
        if not html_script.exists():
            return False

        # Read template from script
        import importlib.util
        spec = importlib.util.spec_from_file_location("pdf_gen", str(html_script))
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        html_content = getattr(mod, "HTML_TEMPLATE", "")
        if not html_content:
            return False

        html_path = root / "SUBMISSION.html"
        html_path.write_text(html_content, encoding="utf-8")

        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                headless=True,
            )
            page = browser.new_page()
            page.set_content(html_content, wait_until="networkidle")
            page.pdf(
                path=str(output_pdf),
                format="A4",
                print_background=True,
                display_header_footer=False,
                margin={"top": "14mm", "bottom": "14mm", "left": "12mm", "right": "12mm"}
            )
            browser.close()
        return True
    except Exception:
        return False


def build_submission(root: Path = Path(".")) -> tuple[Path, Path]:
    final = root / "eval" / "final"
    reports = []
    if final.exists():
        for path in sorted(final.glob("*.json")):
            reports.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
    eval_text = (root / "docs" / "EVAL.md").read_text(encoding="utf-8") if (root / "docs" / "EVAL.md").exists() else "No generated evaluation report."
    design = (root / "docs" / "DESIGN.md").read_text(encoding="utf-8") if (root / "docs" / "DESIGN.md").exists() else ""
    ai_use = (root / "docs" / "AI_USE.md").read_text(encoding="utf-8") if (root / "docs" / "AI_USE.md").exists() else ""
    lines = [
        "# Asterline Assistant — Take-Home Assessment",
        "",
        f"Name: {NAME}",
        f"NIAT ID: {NIAT_ID}",
        f"Repository: {GITHUB_URL}",
        f"LangSmith traces: {LANGSMITH_URL}",
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
        scenarios = data.get("scenarios", [])
        case_results = data.get("case_results", [])
        if scenarios:
            passed = sum(bool(scenario.get("passed")) for scenario in scenarios)
            lines.append(f"{name}: provider={data.get('provider', 'unknown')} scenarios={len(scenarios)} passed={passed}/{len(scenarios)}")
            for metric in ("memory_write_precision", "memory_write_recall", "forget_success", "isolation_leakage", "secret_leak_count", "sensitive_stored_count"):
                value = data.get(metric)
                if isinstance(value, (int, float)):
                    lines.append(f"  {metric}: {value} (aggregate over {len(scenarios)} scenarios)")
            for category, rate in sorted(data.get("category_pass_rates", {}).items()):
                lines.append(f"  {category} pass rate: {rate}")
        else:
            case_count = data.get("cases", len(case_results))
            lines.append(f"{name}: provider={data.get('provider')} config={data.get('config', 'unknown')} cases={case_count}")
            if data.get("model"):
                lines.append(f"  model: {data['model']}")
            if "no_fallback" in data:
                lines.append(f"  fallback disabled: {bool(data['no_fallback'])}")
            for metric, value in data.get("overall", {}).items():
                if isinstance(value, dict) and "value" in value:
                    lines.append(f"  {metric}: {value['value']} ({value.get('applicable', 0)}/{value.get('total', 0)} applicable)")
                elif isinstance(value, (int, float)):
                    lines.append(f"  {metric}: {value}")
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

    # In the actual repo root, generate rich publication-quality PDF via Chromium
    is_real_repo = (root / "data" / "corpus.jsonl").exists()
    generated = False
    if is_real_repo:
        generated = _generate_rich_pdf(pdf_path, root)

    # In test tmp_path or if Playwright is unavailable, fall back to simple PDF
    if not generated:
        pdf_lines = [re.sub(r"[`*_]", "", line) for line in lines]
        _write_simple_pdf(pdf_path, pdf_lines)

    return md_path, pdf_path
