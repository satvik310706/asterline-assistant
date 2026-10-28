import re
from pathlib import Path

from asterline_assistant.calculator import calculate
from asterline_assistant.corpus import load_corpus
from asterline_assistant.models import EvidencePack, RetrievedChunk


CORPUS = {chunk.chunk_id: chunk for chunk in load_corpus(Path(__file__).parents[1] / "data" / "corpus.jsonl")}


def evidence_for(question: str, chunk_ids: tuple[str, ...]) -> EvidencePack:
    return EvidencePack(query=question, chunks=[RetrievedChunk(chunk=CORPUS[chunk_id]) for chunk_id in chunk_ids])


def test_deterministic_calculator_uses_retrieved_evidence_for_supported_cases() -> None:
    cases = [
        ("My Team workspace needs 25 seats in total. What is the monthly cost including the base plan?", ("pricing:0", "pricing:1"), r"\b529\b"),
        ("What is the total yearly cost of the Team plan with annual billing, before any extra seats?", ("pricing:0",), r"4,741\.44"),
        ("What is the total yearly cost of the Launch plan with annual billing?", ("pricing:0",), r"623\.04"),
        ("A Team customer ingests 17 million run records in a month. What is the overage charge?", ("pricing:0", "pricing:1"), r"\$140\.00"),
        ("A Launch customer ingests 1.5 million run records in a month. What overage do they pay?", ("pricing:0", "pricing:1"), r"\$60\.00"),
        ("An Enterprise customer's monthly uptime was 99.3%. What service credit applies?", ("sla:0",), r"25%"),
        ("An Enterprise customer had 98.9% uptime. What credit applies, and are credits added together?", ("sla:0",), r"50%"),
        ("A Team customer's records leave searchable retention. How many days after ingestion are they permanently deleted from Asterline-managed storage?", ("pricing:1", "retention:0"), r"\b200 days\b"),
        ("A Launch customer's records leave searchable retention. How many days after ingestion are they permanently deleted from Asterline-managed storage?", ("pricing:1", "retention:0"), r"\b41 days\b"),
        ("What is the ingestion rate limit per project on the Launch plan?", ("ingest:1", "release-21:0"), r"300 requests per second per project"),
        ("How long can a Team customer search their run records, and does cold recovery storage extend that window?", ("pricing:1", "retention:0", "retention:1"), r"180 days"),
    ]

    for question, chunk_ids, expected in cases:
        response = calculate(question, evidence_for(question, chunk_ids))
        assert response is not None, question
        assert re.search(expected, response.answer), response.answer
        assert set(response.cited_chunk_ids) <= set(chunk_ids)
        assert response.trace_metadata["provider"] == "calculator"
        assert response.trace_metadata["degraded"] is False


def test_calculator_refuses_unpublished_enterprise_overage() -> None:
    question = "What is the Enterprise overage price per 10,000 records?"
    response = calculate(question, evidence_for(question, ("pricing:1",)))
    assert response is not None
    assert "negotiated" in response.answer.lower()
    assert "numeric overage price" in response.answer.lower()
    assert response.cited_chunk_ids == ["pricing:1"]


def test_calculator_applies_annual_discount_only_to_published_base_plans() -> None:
    question = "Does the annual-billing discount apply to Enterprise, and to which parts of the Launch and Team bills?"
    response = calculate(question, evidence_for(question, ("pricing:0",)))

    assert response is not None
    assert "12%" in response.answer
    assert "Launch and Team base subscriptions only" in response.answer
    assert "does not state an Enterprise annual discount" in response.answer
    assert response.cited_chunk_ids == ["pricing:0"]


def test_calculator_flags_historical_launch_rate_and_keeps_search_retention_distinct() -> None:
    launch_question = "What is the ingestion rate limit per project on the Launch plan?"
    launch = calculate(launch_question, evidence_for(launch_question, ("ingest:1", "release-21:0")))
    assert launch is not None
    assert "300 requests per second per project" in launch.answer
    assert "180" in launch.answer and "superseded" in launch.answer.lower()
    assert set(launch.cited_chunk_ids) == {"ingest:1", "release-21:0"}

    retention_question = "How long can a Team customer search their run records, and does cold recovery storage extend that window?"
    retention = calculate(retention_question, evidence_for(retention_question, ("pricing:1", "retention:0", "retention:1")))
    assert retention is not None
    assert "180 days" in retention.answer
    assert re.search(r"not searchable.{0,60}does not extend searchable retention", retention.answer)
    assert set(retention.cited_chunk_ids) == {"pricing:1", "retention:0", "retention:1"}


def test_calculator_answers_current_launch_rate_without_per_project_wording() -> None:
    question = "What is the current Launch ingestion rate?"
    response = calculate(question, evidence_for(question, ("ingest:1", "release-21:0")))

    assert response is not None
    assert "300 requests per second per project" in response.answer
    assert "180" in response.answer and "superseded" in response.answer.lower()
    assert set(response.cited_chunk_ids) == {"ingest:1", "release-21:0"}


def test_calculator_does_not_cite_chunks_outside_the_evidence_pack() -> None:
    question = "What is the yearly Team base price with annual billing?"
    response = calculate(question, evidence_for(question, ()))
    assert response is None
