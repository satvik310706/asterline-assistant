"""Runtime configuration for the small, reproducible baseline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    config: str = Field(default="baseline", pattern="^(baseline|improved)$")
    data_dir: Path = Path("data")
    corpus_path: Path = Path("data/corpus.jsonl")
    chroma_path: Path = Path("data/chroma")
    fastembed_cache_path: Path = Path(".cache/fastembed")
    sqlite_path: Path = Path("data/asterline.sqlite3")
    llm_provider: str = "auto"
    gemini_model: str = "gemini-2.5-flash"
    ollama_model: str = "llama3.2:3b"
    eval_min_interval_seconds: float = 8.0
    eval_no_fallback: bool = False
    eval_langsmith_project: str = "asterline-eval"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            config=os.getenv("ASTERLINE_CONFIG", "baseline"),
            llm_provider=os.getenv("ASTERLINE_LLM_PROVIDER", "auto"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            eval_min_interval_seconds=float(os.getenv("ASTERLINE_EVAL_MIN_INTERVAL_SECONDS", os.getenv("ASTERLINE_EVAL_DELAY_SECONDS", "8"))),
            eval_no_fallback=os.getenv("ASTERLINE_EVAL_NO_FALLBACK", "0") == "1",
            eval_langsmith_project=os.getenv("ASTERLINE_EVAL_LANGSMITH_PROJECT", "asterline-eval"),
        )

