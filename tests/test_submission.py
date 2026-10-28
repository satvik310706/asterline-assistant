from pathlib import Path

from asterline_assistant.submission import build_submission


def test_build_submission_reads_generated_reports(tmp_path: Path) -> None:
    (tmp_path / "eval" / "final").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "eval" / "final" / "baseline_fake.json").write_text(
        '{"provider":"fake","config":"baseline","cases":1,"overall":{"fact_accuracy":{"value":1.0,"applicable":1,"total":1}}}',
        encoding="utf-8",
    )
    (tmp_path / "docs" / "EVAL.md").write_text("# Eval\n", encoding="utf-8")
    (tmp_path / "docs" / "DESIGN.md").write_text("# Design\n", encoding="utf-8")
    (tmp_path / "docs" / "AI_USE.md").write_text("# AI use\n", encoding="utf-8")
    markdown, pdf = build_submission(tmp_path)
    rendered = markdown.read_text(encoding="utf-8")
    assert "fact_accuracy: 1.0" in rendered
    assert "https://github.com/satvik310706/asterline-assistant" in rendered
    assert pdf.read_bytes().startswith(b"%PDF-1.4")


def test_build_submission_counts_memory_scenarios_and_metrics(tmp_path: Path) -> None:
    (tmp_path / "eval" / "final").mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    (tmp_path / "eval" / "final" / "memory_fake.json").write_text(
        '{"provider":"fake","scenarios":[{"id":"M01","passed":true}],"category_pass_rates":{"isolation":1.0},"memory_write_precision":1.0,"isolation_leakage":0}',
        encoding="utf-8",
    )
    (tmp_path / "docs" / "EVAL.md").write_text("# Eval\n", encoding="utf-8")
    (tmp_path / "docs" / "DESIGN.md").write_text("# Design\n", encoding="utf-8")
    (tmp_path / "docs" / "AI_USE.md").write_text("# AI use\n", encoding="utf-8")

    markdown, _ = build_submission(tmp_path)
    rendered = markdown.read_text(encoding="utf-8")
    assert "memory_fake.json: provider=fake scenarios=1 passed=1/1" in rendered
    assert "memory_write_precision: 1.0" in rendered
    assert "cases=None" not in rendered
