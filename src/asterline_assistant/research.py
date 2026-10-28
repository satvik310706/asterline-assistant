"""Baseline research path: route, retrieve, build evidence, answer."""

from __future__ import annotations

import re

from .llm import LLMProvider
from .memory import MemoryStore
from .models import AnswerDraft, EvidencePack, HistoryMessage, ResearchResponse
from .retrieval import HybridRetriever, tokenize
from .tracing import traced


@traced("router")
def route(question: str, history: list[HistoryMessage] | None = None) -> str:
    """Make follow-ups searchable without asking an LLM to rewrite them."""
    if not history:
        return question.strip()
    previous_user = next((item.content for item in reversed(history) if item.role == "user"), "")
    return f"{previous_user} Follow-up: {question.strip()}" if previous_user else question.strip()


@traced("retriever")
def retrieve(query: str, retriever: HybridRetriever) -> EvidencePack:
    return EvidencePack(query=query, chunks=retriever.search(query, k=5))


def build_evidence(pack: EvidencePack) -> EvidencePack:
    return pack


@traced("answer_writer")
def answer(question: str, evidence: EvidencePack, provider: LLMProvider) -> AnswerDraft:
    query_terms = {term for term in tokenize(question) if len(term) > 2}
    evidence_terms = set(tokenize(" ".join(item.chunk.text for item in evidence.chunks)))
    if query_terms and not query_terms.intersection(evidence_terms):
        return AnswerDraft(answer="I could not find enough evidence in the supplied corpus.", abstained=True)
    return provider.answer(question, evidence)


def run_research(question: str, user_id: str, retriever: HybridRetriever, provider: LLMProvider, memory: MemoryStore, history: list[HistoryMessage] | None = None) -> ResearchResponse:
    query = route(question, history)
    used = [item.content for item in memory.list_active(user_id)]
    pack = build_evidence(retrieve(query, retriever))
    draft = answer(query, pack, provider)
    written: list[str] = []
    match = re.match(r"(?is)^remember(?: that)?\s+(.+)$", question.strip())
    if match and memory.write_explicit(user_id, match.group(1).strip()):
        written.append(match.group(1).strip())
    return ResearchResponse(answer=draft.answer, evidence=pack.chunks, memory_used=used, memory_written=written, abstained=draft.abstained, provider=provider.name, model=provider.model, degraded=draft.degraded, attempts=draft.attempts, error_class=draft.error_class, provider_error=draft.provider_error)

