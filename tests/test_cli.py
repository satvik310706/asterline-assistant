from pathlib import Path
import sys

from asterline_assistant import cli
from asterline_assistant.config import Settings


def test_eval_provider_flag_overrides_environment_default(monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setattr(sys, "argv", ["asterline", "eval", "--provider", "ollama"])
    monkeypatch.setattr(cli.Settings, "from_environment", lambda: Settings(llm_provider="fake"))

    def fake_eval(dataset: Path, settings: Settings, **kwargs) -> Path:
        captured["provider"] = settings.llm_provider
        return Path("eval/results/test.json")

    monkeypatch.setattr(cli, "run_eval", fake_eval)
    cli.main()

    assert captured["provider"] == "ollama"
