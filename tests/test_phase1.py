from pathlib import Path

from asterline_assistant.config import Settings
from asterline_assistant.corpus import EXPECTED_CORPUS_LINES, EXPECTED_CORPUS_SHA256, corpus_sha256, load_corpus
from asterline_assistant.llm import FakeLLM
from asterline_assistant.memory import MemoryStore
from asterline_assistant.models import EvidencePack
from asterline_assistant.retrieval import TOKEN_RE, tokenize
from asterline_assistant.retrieval import HybridRetriever


ROOT = Path(__file__).parents[1]


def test_corpus_integrity() -> None:
    path = ROOT / "data/corpus.jsonl"
    assert sum(1 for _ in path.open(encoding="utf-8")) == EXPECTED_CORPUS_LINES
    assert corpus_sha256(path) == EXPECTED_CORPUS_SHA256
    assert len(load_corpus(path)) == 75


def test_tokenizer_preserves_operational_tokens() -> None:
    tokens = tokenize("INC-2025-12 2.3.1 429 99.92%")
    assert {"inc-2025-12", "2.3.1", "429", "99.92%"} <= set(tokens)


def test_fake_llm_cites_retrieved_chunks() -> None:
    chunks = load_corpus(ROOT / "data/corpus.jsonl")[:1]
    result = FakeLLM().answer("what is this", EvidencePack(query="what is this", chunks=[{"chunk": chunks[0]}]))
    assert chunks[0].chunk_id in result.cited_chunk_ids
    assert f"[{chunks[0].chunk_id} — {chunks[0].title}]" in result.answer


def test_memory_isolation_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    first = MemoryStore(path)
    first.write_explicit("alice", "uses Team")
    first.connection.close()
    second = MemoryStore(path)
    assert [m.content for m in second.list_active("alice")] == ["uses Team"]
    assert second.list_active("bob") == []


def test_memory_filters_secret_and_sensitive_claims(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    assert store.write_explicit("alice", "API key: synthetic-secret") is None
    assert store.write_explicit("alice", "my medical diagnosis is private") is None


def test_hybrid_retrieval_uses_one_entry_per_chunk(tmp_path: Path) -> None:
    chunks = load_corpus(ROOT / "data/corpus.jsonl")
    retriever = HybridRetriever(chunks, tmp_path / "chroma")
    results = retriever.search("INC-2025-12 duplicate rows", k=5)
    assert len(results) == 5
    assert any(item.chunk.chunk_id == "incident-12:0" for item in results)

