"""Deterministic smoke evaluation for the baseline configuration."""

from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .corpus import load_corpus, validate_corpus_integrity
from .llm import make_provider
from .memory import MemoryStore
from .research import run_research
from .retrieval import HybridRetriever


def _matches(pattern: str, text: str) -> bool:
    try:
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    except re.error:
        return pattern.casefold() in text.casefold()


def run(dataset: Path, settings: Settings) -> Path:
    validate_corpus_integrity(settings.corpus_path)
    retriever = HybridRetriever(load_corpus(settings.corpus_path), settings.chroma_path)
    provider = make_provider("fake", settings.gemini_model, settings.ollama_model)
    memory = MemoryStore(settings.sqlite_path)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    scores = {"retrieval_recall_at_5": [], "mrr": [], "fact_match": [], "citation_validity": [], "abstention_accuracy": []}
    cache_path = Path("eval/results/cache/responses.json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for row in rows:
        cache_key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if cache_key in cache:
            from .models import ResearchResponse
            response = ResearchResponse.model_validate(cache[cache_key])
        else:
            response = run_research(row["question"], "eval", retriever, provider, memory)
            cache[cache_key] = response.model_dump(mode="json")
        retrieved = [item.chunk.chunk_id for item in response.evidence]
        gold = row.get("gold_chunks", [])
        scores["retrieval_recall_at_5"].append(float(not gold or any(chunk in retrieved for chunk in gold)))
        scores["mrr"].append(next((1 / (retrieved.index(chunk) + 1) for chunk in gold if chunk in retrieved), 0.0))
        scores["fact_match"].append(float(all(_matches(pattern, response.answer) for pattern in row.get("gold_facts", [])) and not any(_matches(pattern, response.answer) for pattern in row.get("forbidden_facts", []))))
        scores["citation_validity"].append(float(all(chunk in retrieved for chunk in re.findall(r"\[([^\s—]+)", response.answer))))
        expected_abstain = row.get("expected") in {"abstain", "clarify"}
        scores["abstention_accuracy"].append(float(response.abstained == expected_abstain))
    result = {key: sum(values) / len(values) if values else 0.0 for key, values in scores.items()}
    result["config"] = settings.config
    result["cases"] = len(rows)
    output = Path("eval/results")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{settings.config}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return path

