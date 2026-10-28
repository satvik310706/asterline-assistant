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


def run_research(question: str, user_id: str, retriever: HybridRetriever, provider: LLMProvider, memory: MemoryStore, history: list[HistoryMessage] | None = None, prepared_query: str | None = None, prepared_evidence: EvidencePack | None = None) -> ResearchResponse:
    command = question.strip()
    if re.match(r"(?i)^(list|show) (my )?memories$", command):
        records = memory.list_active(user_id)
        answer_text = "Saved context:\n" + "\n".join(f"- {record.content}" for record in records) if records else "You have no saved context."
        return ResearchResponse(answer=answer_text, evidence=[], provider=provider.name, model=provider.model, memory_used=[record.content for record in records])
    if re.match(r"(?i)^forget (everything|all)( about me)?$", command):
        count = memory.forget_all(user_id)
        return ResearchResponse(answer=f"Forgot {count} saved memor{'y' if count == 1 else 'ies'}.", evidence=[], provider=provider.name, model=provider.model)
    forget_match = re.match(r"(?is)^forget(?: that)?\s+(.+)$", command)
    if forget_match:
        target = forget_match.group(1).strip()
        count = memory.forget(user_id, target)
        answer_text = f"Forgot {count} matching saved memor{'y' if count == 1 else 'ies'}." if count else "I could not find one unambiguous saved memory matching that request."
        return ResearchResponse(answer=answer_text, evidence=[], provider=provider.name, model=provider.model)
    correction = re.match(r"(?is)^correct(?: that)?\s+(.+?)\s+(?:to|->|instead say)\s+(.+)$", command)
    if correction:
        record = memory.correct(user_id, correction.group(1).strip(), correction.group(2).strip())
        answer_text = f"Corrected saved context to: {record.content}" if record else "I could not identify one unambiguous saved memory to correct."
        return ResearchResponse(answer=answer_text, evidence=[], provider=provider.name, model=provider.model, memory_written=[record.content] if record else [])
    query = prepared_query if prepared_query is not None else route(question, history)
    used = [item.content for item in memory.list_active(user_id)]
    pack = prepared_evidence if prepared_evidence is not None else build_evidence(retrieve(query, retriever))
    draft = answer(query, pack, provider)
    written: list[str] = []
    match = re.match(r"(?is)^remember(?: that)?\s+(.+)$", question.strip())
    if match:
        record = memory.write_explicit(user_id, match.group(1).strip())
        if record:
            written.append(record.content)
        else:
            return ResearchResponse(answer="I cannot save secrets or sensitive personal information.", evidence=pack.chunks, memory_used=used, provider=provider.name, model=provider.model)
    return ResearchResponse(answer=draft.answer, evidence=pack.chunks, memory_used=used, memory_written=written, abstained=getattr(draft, "abstained", False), provider=provider.name, model=provider.model, degraded=getattr(draft, "degraded", False), attempts=getattr(draft, "attempts", 0), error_class=getattr(draft, "error_class", None), primary_error=getattr(draft, "primary_error", None), fallback_error=getattr(draft, "fallback_error", None), provider_error=getattr(draft, "provider_error", False))

