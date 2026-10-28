from pathlib import Path

from asterline_assistant.graph import _build_graph, run_improved_research
from asterline_assistant.llm import FakeLLM
from asterline_assistant.memory import MemoryStore
from asterline_assistant.models import CorpusChunk, RetrievedChunk


class OneChunkRetriever:
    chunks = []

    def __init__(self) -> None:
        self.chunk = CorpusChunk(
            chunk_id="demo:0", doc_id="demo", title="Demo source", category="demo", owner="test",
            source_url="test://demo", published="20260101", version="1", text="Launch allows 300 requests per second.",
        )
        self.chunks = [self.chunk]

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        return [RetrievedChunk(chunk=self.chunk)]


def test_improved_graph_has_seven_named_nodes() -> None:
    assert set(_build_graph().nodes) >= {"router", "memory_reader", "retriever", "evidence_analyst", "answer_writer", "verifier", "memory_writer"}


def test_improved_graph_returns_cited_answer(tmp_path: Path) -> None:
    response = run_improved_research("What is the Launch rate?", "alice", OneChunkRetriever(), FakeLLM(), MemoryStore(tmp_path / "memory.sqlite3"))
    assert "demo:0" in response.answer
    assert response.provider == "fake"


def test_improved_graph_emits_completed_node_timings(tmp_path: Path) -> None:
    events: list[tuple[str, float]] = []
    run_improved_research(
        "What is the Launch rate?", "alice", OneChunkRetriever(), FakeLLM(),
        MemoryStore(tmp_path / "memory.sqlite3"), on_step=lambda node, elapsed: events.append((node, elapsed)),
    )
    assert [node for node, _ in events] == ["router", "memory_reader", "retriever", "evidence_analyst", "answer_writer", "verifier", "memory_writer"]
    assert all(elapsed >= 0 for _, elapsed in events)


def test_chitchat_skips_retrieval_and_llm(tmp_path: Path) -> None:
    events: list[tuple[str, float]] = []
    provider = FakeLLM()
    response = run_improved_research(
        "hi", "alice", OneChunkRetriever(), provider, MemoryStore(tmp_path / "memory.sqlite3"),
        on_step=lambda node, elapsed: events.append((node, elapsed)),
    )
    assert "What would you like to know?" in response.answer
    assert response.evidence == []
    assert response.provider == "rules"
    assert response.model == "deterministic"
    assert provider.calls == 0
    assert [node for node, _ in events] == ["router", "answer_writer"]
