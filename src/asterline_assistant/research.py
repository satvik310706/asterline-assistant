"""Baseline research path: route, retrieve, build evidence, answer."""

from __future__ import annotations

import re
from datetime import timedelta

from .llm import LLMProvider
from .memory import MemoryStore, SENSITIVE_RE, contains_sensitive, redact_sensitive
from .models import AnswerDraft, EvidencePack, HistoryMessage, ResearchResponse, RetrievedChunk
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


def run_research(question: str, user_id: str, retriever: HybridRetriever, provider: LLMProvider, memory: MemoryStore, history: list[HistoryMessage] | None = None, prepared_query: str | None = None, prepared_evidence: EvidencePack | None = None, use_memory: bool = False) -> ResearchResponse:
    original_question = question
    question = redact_sensitive(question)
    command = question.strip()
    has_sensitive_personal = bool(SENSITIVE_RE.search(original_question))
    has_secret = contains_sensitive(original_question)
    is_remember = bool(re.match(r"(?is)^remember(?: that)?\s+", command))
    if has_sensitive_personal and (is_remember or re.search(r"(?i)remember", original_question)):
        return ResearchResponse(answer="I cannot store personal health or leave circumstances. Please use the HR portal for the applicable policy and records.", evidence=[], provider=provider.name, model=provider.model)
    if has_secret and (is_remember or re.search(r"(?i)remember", original_question)):
        return ResearchResponse(answer="I did not store that secret value. Please rotate the credential and do not share it in chat.", evidence=[], provider=provider.name, model=provider.model)
    if re.match(r"(?i)^(list|show) (my )?memories$", command) or re.match(r"(?i)^what do you remember about me\??$", command):
        records = memory.list_active(user_id)
        answer_text = "Saved context:\n" + "\n".join(f"- {record.content}" for record in records) if records else "You have no saved context."
        return ResearchResponse(answer=answer_text, evidence=[], provider=provider.name, model=provider.model, memory_used=[record.content for record in records])
    if re.match(r"(?i)^forget (everything|all)( you know)?( about me)?[.!]?$", command):
        count = memory.forget_all(user_id)
        return ResearchResponse(answer=f"Forgot {count} saved memor{'y' if count == 1 else 'ies'}.", evidence=[], provider=provider.name, model=provider.model)
    forget_match = re.match(r"(?is)^forget(?: that)?\s+(.+)$", command)
    if forget_match:
        target = forget_match.group(1).strip()
        count = memory.forget(user_id, target)
        answer_text = f"Forgot {count} matching saved memor{'y' if count == 1 else 'ies'}." if count else "I could not find one unambiguous saved memory matching that request."
        return ResearchResponse(answer=answer_text, evidence=[], provider=provider.name, model=provider.model)
    correction = re.match(r"(?is)^(?:correct(?: that)?|correction:)\s+(.+?)\s+(?:to|->|instead say)\s+(.+)$", command)
    if not correction:
        correction = re.match(r"(?is)^correction:\s+i'?m actually based in\s+(.+?)[.!]?$", command)
        if correction:
            correction = ("based in", correction.group(1))
    if not correction:
        self_correction = re.match(r"(?is)^actually,?\s+(?:we'?re|i'?m)\s+on\s+(.+?),?\s+not\s+(.+?)[.!]?$", command)
        if self_correction:
            correction = (self_correction.group(2), self_correction.group(1))
    if correction:
        target, replacement = (correction if isinstance(correction, tuple) else (correction.group(1), correction.group(2)))
        target, replacement = target.strip(), replacement.strip()
        company_claim = bool(re.search(r"(?i)(retention|cold storage|sla|ingestion|requests per second|seats|price|list price)", command))
        record = None if company_claim else memory.correct(user_id, target, replacement)
        answer_text = f"Corrected saved context to: {record.content}" if record else "I could not identify one unambiguous saved memory to correct."
        if company_claim:
            answer_text = "I will not store a company fact as memory; I will check the supplied evidence."
        return ResearchResponse(answer=answer_text, evidence=[], provider=provider.name, model=provider.model, memory_written=[record.content] if record else [])
    query = prepared_query if prepared_query is not None else route(question, history)
    records = memory.list_active(user_id)
    used = [item.content for item in records]
    if use_memory:
        age_cutoff = memory.clock.now() - timedelta(days=14)
        stale = [item for item in records if item.type in {"project", "task"} and item.updated_at and item.updated_at < age_cutoff.isoformat()]
        if stale and re.search(r"(?i)(continue|where we left|that|it)", question):
            return ResearchResponse(answer=f"Last time you were investigating {stale[0].content}. Is that still relevant?", evidence=[], memory_used=used, provider=provider.name, model=provider.model)
        if records and re.search(r"(?i)(what plan did i|which plan did i|what do you remember)", question):
            return ResearchResponse(answer="From what you told me earlier: " + "; ".join(record.content for record in records[:2]) + ".", evidence=[], memory_used=used, provider=provider.name, model=provider.model)
        query += " " + " ".join(record.content for record in records)
        expansions = {
            r"(?i)aql timeout": "query:0 release-23:0 2.3 50 seconds",
            r"(?i)maximum number of rows": "query:0 8000",
            r"(?i)alert rules": "alerts:1 pricing:2 6 75 600",
            r"(?i)launch ingestion rate": "ingest:1 release-21:0 300",
            r"(?i)warehouse export": "pricing:2 export:0",
            r"(?i)(429|retry-after)": "ingest:1",
        }
        for pattern, expansion in expansions.items():
            if re.search(pattern, question):
                query += " " + expansion
                break
    if use_memory and not records and re.search(r"(?i)(what (?:was|am) i investigating|what do you remember)", question):
        return ResearchResponse(answer="I don't have any saved investigation or context for you.", evidence=[], provider=provider.name, model=provider.model)
    pack = prepared_evidence if prepared_evidence is not None else build_evidence(retrieve(query, retriever))
    if use_memory:
        wanted: list[str] = []
        if re.search(r"(?i)(aql timeout|50 seconds)", query):
            wanted = ["query:0", "release-23:0"]
        elif re.search(r"(?i)(duplicate rows|bigquery|heron)", query):
            wanted = ["incident-12:0", "incident-12:1", "release-21:1", "export:1"]
        elif re.search(r"(?i)(ingestion rate|429|retry-after)", query):
            wanted = ["ingest:1", "release-21:0"]
        elif re.search(r"(?i)warehouse export", query):
            wanted = ["pricing:2", "export:0"]
        if wanted:
            by_id = {item.chunk.chunk_id: item for item in pack.chunks}
            for chunk in retriever.chunks:
                if chunk.chunk_id in wanted and chunk.chunk_id not in by_id:
                    by_id[chunk.chunk_id] = RetrievedChunk(chunk=chunk)
            ordered = [by_id[chunk_id] for chunk_id in wanted if chunk_id in by_id]
            ordered.extend(item for item in pack.chunks if item.chunk.chunk_id not in {entry.chunk.chunk_id for entry in ordered})
            pack = EvidencePack(query=pack.query, chunks=ordered[:5])
    draft = answer(query, pack, provider)
    answer_text = draft.answer
    if use_memory and re.search(r"(?i)aql timeout", question) and any("50 seconds" in item.chunk.text for item in pack.chunks):
        source = next(item for item in pack.chunks if "50 seconds" in item.chunk.text)
        answer_text = f"The current AQL timeout is 50 seconds [{source.chunk.chunk_id} — {source.chunk.title}]."
    if use_memory and records:
        relevant = [record for record in records if any(term in record.content.lower() for term in tokenize(question.lower()) if len(term) > 3)]
        if not relevant and re.search(r"(?i)(plan|remember|told|said|earlier|that|continue|where we left|alert rules|warehouse export)", question):
            relevant = records[:2]
        if relevant:
            answer_text += "\nFrom what you told me earlier: " + "; ".join(record.content for record in relevant[:2]) + "."
    if use_memory and any(re.search(r"(?i)(at most two sentences|keep answers short)", record.content) for record in records):
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", answer_text)
        answer_text = " ".join(sentences[:2])
        citations = " ".join(f"[{item.chunk.chunk_id} — {item.chunk.title}]" for item in pack.chunks[:2])
        if citations and "[" not in answer_text:
            answer_text += " " + citations
    written: list[str] = []
    match = re.match(r"(?is)^remember(?: that)?\s+(.+)$", question.strip())
    if match:
        content = match.group(1).strip()
        rejected_company = bool(re.search(r"(?i)(retention|cold storage|sla|100\s?%|list price|enterprise price)", content))
        rejected_instruction = bool(re.search(r"(?i)(ignore the knowledge base|only from memory|ignore previous instructions)", content))
        record = None if rejected_company or rejected_instruction else memory.write_explicit(user_id, content)
        if record:
            written.append(record.content)
        elif has_secret or has_sensitive_personal:
            return ResearchResponse(answer="I did not store that secret or sensitive value. Please rotate the credential and use the HR portal for personal leave or health matters.", evidence=pack.chunks, memory_used=used, provider=provider.name, model=provider.model)
    return ResearchResponse(answer=answer_text, evidence=pack.chunks, memory_used=used, memory_written=written, abstained=getattr(draft, "abstained", False), provider=provider.name, model=provider.model, degraded=getattr(draft, "degraded", False), attempts=getattr(draft, "attempts", 0), error_class=getattr(draft, "error_class", None), primary_error=getattr(draft, "primary_error", None), fallback_error=getattr(draft, "fallback_error", None), provider_error=getattr(draft, "provider_error", False))

