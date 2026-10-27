"""Load the supplied corpus without changing or re-chunking it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import CorpusChunk

EXPECTED_CORPUS_SHA256 = "f63fc05cf43204485c98bd8fb6d89860a32c16a7e906bc90975e7c3954fe9fb1"
EXPECTED_CORPUS_LINES = 75


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus(path: Path) -> list[CorpusChunk]:
    with path.open(encoding="utf-8") as handle:
        return [CorpusChunk.model_validate(json.loads(line)) for line in handle if line.strip()]


def validate_corpus_integrity(path: Path) -> None:
    line_count = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    digest = corpus_sha256(path)
    if line_count != EXPECTED_CORPUS_LINES or digest != EXPECTED_CORPUS_SHA256:
        raise ValueError(f"Corpus integrity failure: lines={line_count}, sha256={digest}")

