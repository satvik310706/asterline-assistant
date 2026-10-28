from pathlib import Path

from asterline_assistant.config import Settings
from asterline_assistant.corpus import EXPECTED_CORPUS_LINES, EXPECTED_CORPUS_SHA256, corpus_sha256, load_corpus
from asterline_assistant.llm import FakeLLM
from asterline_assistant.memory import MemoryStore
from asterline_assistant.models import EvidencePack
from asterline_assistant.retrieval import TOKEN_RE, tokenize
from asterline_assistant.eval_runner import abstention_pass, fact_pass, superseded_pass
from asterline_assistant.models import HistoryMessage
from asterline_assistant.research import route
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


def test_eval_matching_allows_forbidden_fact_only_when_superseded() -> None:
    answer = "The older value was 60 days; the current policy is 20 days and supersedes it."
    assert fact_pass(answer, [r"20 days"], [r"60 days"])
    assert superseded_pass(answer, [r"60 days"])


def test_eval_matching_rejects_unqualified_forbidden_fact() -> None:
    assert not fact_pass("Cold recovery lasts 60 days.", [r"20 days"], [r"60 days"])
    assert not superseded_pass("Cold recovery lasts 60 days.", [r"60 days"])


def test_eval_abstention_requires_decline_and_gold_wording() -> None:
    assert abstention_pass("Enterprise pricing is negotiated and not published.", [r"negotiated"], [])
    assert not abstention_pass("Enterprise costs $999.", [r"not published"], [r"\$999"])


def test_router_rewrites_follow_up_from_history() -> None:
    query = route("And what was it before that?", [HistoryMessage(role="user", content="What is the current AQL timeout?")])
    assert "current AQL timeout" in query
    assert "before that" in query

