"""The selectable improved research graph.

The baseline remains in :mod:`research`; this graph adds explicit, inspectable
agent boundaries without moving policy decisions into prompts.
"""

from __future__ import annotations

import re
import hashlib
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .llm import LLMProvider
from .memory import MemoryStore
from .models import EvidencePack, HistoryMessage, MemoryRecord, ResearchResponse
from .research import build_evidence, retrieve, route, run_research
from .retrieval import HybridRetriever
from .tracing import clear_trace_context, set_trace_context, traced


class RouterResult(BaseModel):
    intent: str
    standalone_query: str
    sub_questions: list[str] = Field(default_factory=list)


class MemoryReaderResult(BaseModel):
    records: list[MemoryRecord] = Field(default_factory=list)
    label: str = "UNVERIFIED USER CONTEXT"


class EvidenceAssessment(BaseModel):
    evidence: EvidencePack
    labels: dict[str, str] = Field(default_factory=dict)
    sufficiency: str = "unanswerable"
    clarification: str | None = None


class VerifierResult(BaseModel):
    valid: bool
    cited_ids: list[str] = Field(default_factory=list)
    reason: str | None = None


class GraphState(TypedDict, total=False):
    question: str
    user_id: str
    history: list[HistoryMessage]
    retriever: HybridRetriever
    provider: LLMProvider
    memory: MemoryStore
    router: RouterResult
    memory_reader: MemoryReaderResult
    evidence: EvidencePack
    assessment: EvidenceAssessment
    response: ResearchResponse
    verifier: VerifierResult


def _intent(question: str, history: list[HistoryMessage]) -> str:
    text = question.strip().lower()
    if re.match(r"^(remember|save)\b", text):
        return "remember"
    if re.match(r"^correct\b|^actually\b|^correction:", text):
        return "correct"
    if re.match(r"^forget\b", text):
        return "forget"
    if re.match(r"^(list|show) (my )?memories|what do you remember", text):
        return "list"
    if history and re.search(r"\b(it|that|those|what about|and|also)\b", text):
        return "followup"
    if text in {"hi", "hello", "thanks", "thank you"}:
        return "chitchat"
    return "question"


@traced("router")
def router_agent(state: GraphState) -> GraphState:
    query = route(state["question"], state.get("history", []))
    result = RouterResult(intent=_intent(state["question"], state.get("history", [])), standalone_query=query, sub_questions=[query])
    return {"router": result}


@traced("memory_reader")
def memory_reader_agent(state: GraphState) -> GraphState:
    records = state["memory"].list_active(state["user_id"])
    return {"memory_reader": MemoryReaderResult(records=records)}


@traced("retriever")
def retriever_agent(state: GraphState) -> GraphState:
    query = state["router"].standalone_query
    records = state["memory_reader"].records
    # Memory can disambiguate a query, but it is never placed in the evidence pack.
    if records:
        query += " " + " ".join(record.content for record in records[:3])
    return {"evidence": build_evidence(retrieve(query, state["retriever"]))}


@traced("evidence_analyst")
def evidence_analyst_agent(state: GraphState) -> GraphState:
    pack = state["evidence"]
    labels: dict[str, str] = {}
    for item in pack.chunks:
        text = item.chunk.text.lower()
        labels[item.chunk.chunk_id] = "SUPERSEDED" if re.search(r"\b(supersedes|replaces|earlier|older|outdated)\b", text) else "CURRENT"
    sufficient = "answerable" if pack.chunks else "unanswerable"
    return {"assessment": EvidenceAssessment(evidence=pack, labels=labels, sufficiency=sufficient)}


@traced("answer_writer")
def answer_writer_agent(state: GraphState) -> GraphState:
    response = run_research(
        state["question"], state["user_id"], state["retriever"], state["provider"], state["memory"],
        history=state.get("history", []), prepared_query=state["router"].standalone_query,
        prepared_evidence=state["assessment"].evidence, use_memory=True,
    )
    return {"response": response}


@traced("verifier")
def verifier_agent(state: GraphState) -> GraphState:
    response = state["response"]
    known = {item.chunk.chunk_id for item in state["evidence"].chunks}
    cited = re.findall(r"\[([^\s—]+)\s+—\s+[^\]]+\]", response.answer)
    valid = all(chunk_id in known for chunk_id in cited)
    result = VerifierResult(valid=valid, cited_ids=cited, reason=None if valid else "citation outside evidence pack")
    if not valid:
        response.answer = "I could not verify the cited support in the supplied corpus."
        response.abstained = True
    return {"response": response, "verifier": result}


@traced("memory_writer")
def memory_writer_agent(state: GraphState) -> GraphState:
    # run_research already applies the tested policy gate and writes provenance.
    return {"response": state["response"]}


def _build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("router", router_agent)
    graph.add_node("memory_reader", memory_reader_agent)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("evidence_analyst", evidence_analyst_agent)
    graph.add_node("answer_writer", answer_writer_agent)
    graph.add_node("verifier", verifier_agent)
    graph.add_node("memory_writer", memory_writer_agent)
    graph.set_entry_point("router")
    graph.add_edge("router", "memory_reader")
    graph.add_edge("memory_reader", "retriever")
    graph.add_edge("retriever", "evidence_analyst")
    graph.add_edge("evidence_analyst", "answer_writer")
    graph.add_edge("answer_writer", "verifier")
    graph.add_edge("verifier", "memory_writer")
    graph.add_edge("memory_writer", END)
    return graph.compile()


def run_improved_research(question: str, user_id: str, retriever: HybridRetriever, provider: LLMProvider, memory: MemoryStore, history: list[HistoryMessage] | None = None) -> ResearchResponse:
    set_trace_context({
        "intent": "question",
        "config": "improved",
        "provider": provider.name,
        "model": provider.model,
        "attempts": 0,
        "degraded": False,
        "user_id_hash": hashlib.sha256(user_id.encode()).hexdigest()[:16],
        "session_id": "interactive",
    })
    try:
        result = _build_graph().invoke({"question": question, "user_id": user_id, "history": history or [], "retriever": retriever, "provider": provider, "memory": memory})
        return result["response"]
    finally:
        clear_trace_context()
