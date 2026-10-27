"""Runtime configuration for the small, reproducible baseline."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    config: str = Field(default="baseline", pattern="^(baseline|improved)$")
    data_dir: Path = Path("data")
    corpus_path: Path = Path("data/corpus.jsonl")
    chroma_path: Path = Path("data/chroma")
    sqlite_path: Path = Path("data/asterline.sqlite3")
    llm_provider: str = "auto"
    gemini_model: str = "gemini-2.5-flash"
    ollama_model: str = "llama3.2:3b"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            config=os.getenv("ASTERLINE_CONFIG", "baseline"),
            llm_provider=os.getenv("ASTERLINE_LLM_PROVIDER", "auto"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        )

