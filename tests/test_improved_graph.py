from pathlib import Path

from asterline_assistant.graph import RouterResult, _build_graph, _delegated_authority_answer, _supersession_answer, answer_writer_agent, evidence_analyst_agent, retriever_agent, run_improved_research
from asterline_assistant.corpus import load_corpus
from asterline_assistant.llm import AnswerDraft, FakeLLM, LLMProvider
from asterline_assistant.memory import MemoryStore
from asterline_assistant.models import CorpusChunk, EvidencePack, RetrievedChunk


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


def test_role_and_corpus_inventory_question_uses_deterministic_route(tmp_path: Path) -> None:
    provider = FakeLLM()
    question = "hi, what is your role here? and how many chunks do you have"
    response = run_improved_research(
        question, "alice", OneChunkRetriever(), provider, MemoryStore(tmp_path / "memory.sqlite3"),
    )
    assert "research assistant" in response.answer.lower()
    assert "1 chunk" in response.answer.lower()
    assert response.provider == "rules"
    assert provider.calls == 0


def test_improved_arithmetic_uses_evidence_backed_calculator_without_llm(tmp_path: Path) -> None:
    source_chunks = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}

    class PricingRetriever:
        chunks = list(source_chunks.values())

        def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
            return [RetrievedChunk(chunk=source_chunks[chunk_id]) for chunk_id in ("pricing:0", "pricing:1")]

    provider = FakeLLM()
    response = run_improved_research(
        "My Team workspace needs 25 seats in total. What is the monthly cost including the base plan?",
        "alice", PricingRetriever(), provider, MemoryStore(tmp_path / "memory.sqlite3"),
    )

    assert "$529 per month" in response.answer
    assert "[pricing:0 — Plans and billing]" in response.answer
    assert "[pricing:1 — Plans and billing]" in response.answer
    assert response.provider == "calculator"
    assert response.model == "deterministic"
    assert provider.calls == 0


def test_improved_graph_answers_standard_launch_rate_without_llm(tmp_path: Path) -> None:
    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}

    class IngestionRetriever:
        chunks = list(corpus.values())

        def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
            return [RetrievedChunk(chunk=corpus[chunk_id]) for chunk_id in ("ingest:1", "release-21:0")]

    provider = FakeLLM()
    response = run_improved_research(
        "What is the current Launch ingestion rate?", "alice", IngestionRetriever(), provider,
        MemoryStore(tmp_path / "memory.sqlite3"),
    )

    assert "300 requests per second per project" in response.answer
    assert "180" in response.answer and "superseded" in response.answer.lower()
    assert response.provider == "calculator"
    assert provider.calls == 0


def test_incident_id_question_is_answered_from_retrieved_sources_without_llm(tmp_path: Path) -> None:
    from asterline_assistant.eval_runner import citation_pairs

    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}
    incident_ids = ("incident-08:0", "incident-08:1", "incident-08:2", "release-20:1")

    class IncidentRetriever:
        chunks = list(corpus.values())

        def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
            return [RetrievedChunk(chunk=corpus[chunk_id]) for chunk_id in incident_ids]

    provider = FakeLLM()
    response = run_improved_research(
        "What happened in INC-2025-08?", "alice", IncidentRetriever(), provider,
        MemoryStore(tmp_path / "memory.sqlite3"),
    )
    cited = {chunk_id for chunk_id, _ in citation_pairs(response.answer)}

    assert "32%" in response.answer and "1.4 million" in response.answer
    assert "consumer heartbeat configuration mismatch" in response.answer
    assert cited == set(incident_ids)
    assert response.provider == "rules"
    assert provider.calls == 0


def test_verifier_reports_the_actual_grounding_failure() -> None:
    from asterline_assistant.graph import verifier_agent
    from asterline_assistant.models import ResearchResponse

    response = ResearchResponse(answer="It is 20 days.", evidence=[], provider="gemini", model="test")
    result = verifier_agent({
        "response": response,
        "router": RouterResult(intent="question", standalone_query="q"),
        "evidence": EvidencePack(query="q", chunks=[]),
        "repair_count": 1,
    })
    final = result["response"]

    assert final.error_class == "VerifierFailure"
    assert "numeric claim absent from cited evidence and calculator output" in final.answer


def test_improved_graph_handles_natural_forget_before_evidence_abstention(tmp_path: Path) -> None:
    memory = MemoryStore(tmp_path / "memory.sqlite3")
    memory.write_explicit("alice", "I am investigating BigQuery duplicates on the Team plan.")
    response = run_improved_research(
        "forget that I am investigating BigQuery duplicates", "alice", OneChunkRetriever(), FakeLLM(), memory,
    )

    assert "forgot" in response.answer.lower()
    assert memory.list_active("alice") == []


def test_evidence_analyst_marks_only_explicitly_superseded_same_topic_chunks() -> None:
    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}
    examples = [
        ("What is the current cold recovery period?", "retention:0", "security:1"),
        ("What is the current on-call stipend?", "handbook:1", "oncall:0"),
        ("What is the current AQL timeout?", "query:0", "release-22:2"),
    ]
    for question, current_id, old_id in examples:
        pack = EvidencePack(query=question, chunks=[RetrievedChunk(chunk=corpus[old_id]), RetrievedChunk(chunk=corpus[current_id])])
        result = evidence_analyst_agent({"question": question, "evidence": pack})["assessment"]
        assert result.labels[current_id] == "CURRENT"
        assert result.labels[old_id] == "SUPERSEDED"
        assert result.sufficiency == "answerable"

    severity_question = "Which document defines SEV-1 and SEV-2 status-update deadlines?"
    severity_pack = EvidencePack(query=severity_question, chunks=[
        RetrievedChunk(chunk=corpus["oncall:0"]),
        RetrievedChunk(chunk=corpus["oncall:1"]),
        RetrievedChunk(chunk=corpus["handbook:1"]),
    ])
    severity = evidence_analyst_agent({"question": severity_question, "evidence": severity_pack})["assessment"]
    assert severity.labels["oncall:0"] == "SUPERSEDED"
    assert severity.labels["oncall:1"] == "CURRENT"


def test_evidence_analyst_respects_requested_historical_version() -> None:
    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}
    pack = EvidencePack(query="What was the AQL timeout in version 2.2?", chunks=[RetrievedChunk(chunk=corpus["query:0"]), RetrievedChunk(chunk=corpus["release-22:2"])])
    result = evidence_analyst_agent({"question": pack.query, "evidence": pack})["assessment"]
    assert result.labels["release-22:2"] == "CURRENT"
    assert result.labels["query:0"] == "NOT_APPLICABLE"


def test_evidence_analyst_marks_conflicting_plan_scope_not_applicable() -> None:
    chunk = CorpusChunk(
        chunk_id="explorer:team-enterprise", doc_id="explorer", title="Explorer retention",
        category="product", owner="test", source_url="test://explorer", published="20260310",
        version="2.3", text="Run Explorer timeline retention is 14 days on Team and 60 days on Enterprise.",
    )
    pack = EvidencePack(query="What is Run Explorer timeline retention on Launch?", chunks=[RetrievedChunk(chunk=chunk)])
    result = evidence_analyst_agent({"question": pack.query, "evidence": pack})["assessment"]

    assert result.labels[chunk.chunk_id] == "NOT_APPLICABLE"
    assert result.sufficiency == "unanswerable"


def test_retriever_follows_explicit_plan_retention_reference_when_top_k_misses_table() -> None:
    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}

    class RetentionRetriever:
        chunks = list(corpus.values())

        def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
            return [RetrievedChunk(chunk=corpus[chunk_id]) for chunk_id in ("retention:0", "retention:1", "security:1", "pricing:0", "onboarding:2")]

    question = "A Team customer's records leave searchable retention. How many days after ingestion are they permanently deleted?"
    result = retriever_agent({
        "question": question,
        "router": RouterResult(intent="question", standalone_query=question),
        "retriever": RetentionRetriever(),
    })["evidence"]

    assert "retention:0" in {item.chunk.chunk_id for item in result.chunks}
    assert "pricing:1" in {item.chunk.chunk_id for item in result.chunks}


def test_answer_writer_uses_current_source_for_explicitly_resolved_conflict() -> None:
    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}
    question = "How long do raw run records stay in cold recovery storage after searchable retention ends?"
    evidence = EvidencePack(query=question, chunks=[RetrievedChunk(chunk=corpus["security:1"]), RetrievedChunk(chunk=corpus["retention:0"])])
    assessment = evidence_analyst_agent({"question": question, "evidence": evidence})["assessment"]
    answer_state = answer_writer_agent({
        "question": question,
        "router": RouterResult(intent="question", standalone_query=question),
        "assessment": assessment,
    })
    answer = answer_state["response"].answer

    assert "20 days" in answer
    assert "60 days" in answer or "60-day" in answer
    assert "superseded" in answer.lower()
    assert "[retention:0 — Data retention policy]" in answer
    assert "[security:1 — Security overview]" in answer
    assert answer_state["response"].provider == "rules"


def test_answer_writer_flags_superseded_oncall_stipend_deterministically() -> None:
    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}
    question = "What is the weekly on-call stipend?"
    evidence = EvidencePack(query=question, chunks=[RetrievedChunk(chunk=corpus["oncall:0"]), RetrievedChunk(chunk=corpus["handbook:1"])])
    assessment = evidence_analyst_agent({"question": question, "evidence": evidence})["assessment"]
    result = answer_writer_agent({
        "question": question,
        "router": RouterResult(intent="question", standalone_query=question),
        "assessment": assessment,
    })["response"]

    assert "$175" in result.answer
    assert "$140" in result.answer
    assert "superseded" in result.answer.lower()
    assert "[handbook:1 — Employee handbook]" in result.answer
    assert "[oncall:0 — On-call runbook]" in result.answer


def test_conflict_resolver_handles_a_user_quoting_the_old_stipend() -> None:
    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}
    question = "The on-call runbook lists $140 per week. Is that what engineers receive now?"
    evidence = EvidencePack(query=question, chunks=[RetrievedChunk(chunk=corpus["oncall:0"]), RetrievedChunk(chunk=corpus["handbook:1"])])
    assessment = evidence_analyst_agent({"question": question, "evidence": evidence})["assessment"]
    result = answer_writer_agent({
        "question": question,
        "router": RouterResult(intent="question", standalone_query=question),
        "assessment": assessment,
    })["response"]

    assert "$175" in result.answer
    assert "$140" in result.answer
    assert "superseded" in result.answer.lower()
    assert "[handbook:1 — Employee handbook]" in result.answer
    assert "[oncall:0 — On-call runbook]" in result.answer


def test_conflict_resolver_does_not_answer_stipend_when_question_targets_severity_deadlines() -> None:
    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}
    question = "The handbook now covers on-call pay. Which document defines the SEV-1 and SEV-2 status-update deadlines, and what are they?"
    evidence = EvidencePack(query=question, chunks=[RetrievedChunk(chunk=corpus[chunk_id]) for chunk_id in ("oncall:0", "oncall:1", "handbook:1")])
    assessment = evidence_analyst_agent({"question": question, "evidence": evidence})["assessment"]

    assert assessment.labels["oncall:0"] == "SUPERSEDED"
    assert assessment.labels["oncall:1"] == "CURRENT"
    assert _supersession_answer({"question": question, "assessment": assessment}) is None


def test_answer_writer_follows_corpus_document_ownership_for_severity_deadlines() -> None:
    corpus = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}
    question = "The handbook now covers on-call pay. Which document defines the SEV-1 and SEV-2 status-update deadlines, and what are they?"
    evidence = EvidencePack(query=question, chunks=[RetrievedChunk(chunk=corpus[chunk_id]) for chunk_id in ("oncall:0", "oncall:1", "handbook:1")])
    assessment = evidence_analyst_agent({"question": question, "evidence": evidence})["assessment"]
    assert _delegated_authority_answer({"question": question, "assessment": assessment}) is not None
    result = answer_writer_agent({
        "question": question,
        "router": RouterResult(intent="question", standalone_query=question),
        "assessment": assessment,
    })["response"]

    assert "on-call runbook" in result.answer.lower()
    assert "10 minutes" in result.answer
    assert "45 minutes" in result.answer
    assert "[oncall:1 — On-call runbook]" in result.answer
    assert "[handbook:1 — Employee handbook]" in result.answer
    assert "$175" not in result.answer


def test_verifier_retries_once_then_returns_only_the_verified_answer(tmp_path: Path) -> None:
    class RepairProvider(LLMProvider):
        name = "fake-repair"
        model = "deterministic-test"

        def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
            self.calls += 1
            if self.calls == 1:
                answer = "Launch is limited to 999 requests per second [demo:0 — Demo source]."
            else:
                assert "Repair the previous draft" in question
                answer = "Launch is limited to 300 requests per second [demo:0 — Demo source]."
            return AnswerDraft(answer=answer, cited_chunk_ids=["demo:0"])

    provider = RepairProvider()
    response = run_improved_research(
        "What is the Launch rate?", "alice", OneChunkRetriever(), provider,
        MemoryStore(tmp_path / "repair.sqlite3"),
    )

    assert provider.calls == 2
    assert "300 requests per second" in response.answer
    assert "999" not in response.answer
    assert response.abstained is False
