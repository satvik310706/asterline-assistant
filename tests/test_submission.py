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
    assert "fact_accuracy: 1.0" in markdown.read_text(encoding="utf-8")
    assert pdf.read_bytes().startswith(b"%PDF-1.4")
