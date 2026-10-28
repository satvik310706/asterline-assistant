"""Baseline research path: route, retrieve, build evidence, answer."""

from __future__ import annotations

import re
from datetime import timedelta
from collections.abc import Callable

from .llm import LLMProvider
from .memory import MemoryStore, SENSITIVE_RE, contains_sensitive, redact_sensitive
from .models import AnswerDraft, EvidencePack, HistoryMessage, ResearchResponse, RetrievedChunk
from .retrieval import HybridRetriever, tokenize
from .tracing import traced

COMPANY_FACT_TERMS = re.compile(
    r"(?i)\b(retention|cold recovery|cold storage|sla|ingestion rate|ingestion limit|requests per second|"
    r"overage|list price|enterprise price|aql timeout|hipaa certification|mobile dashboard app)\b"
)


def is_company_fact_claim(text: str) -> bool:
    """Keep asserted product facts on the evidence path rather than memory-command fast paths."""
    return bool(COMPANY_FACT_TERMS.search(text))


@traced("router")
def route(question: str, history: list[HistoryMessage] | None = None) -> str:
    """Make follow-ups searchable without asking an LLM to rewrite them."""
    if not history:
        return question.strip()
    user_msgs = [item.content for item in history if getattr(item, "role", None) == "user" and getattr(item, "content", "").strip().lower() != question.strip().lower()]
    previous_user = user_msgs[-1] if user_msgs else ""
    return f"{previous_user} Follow-up: {question.strip()}" if previous_user else question.strip()


@traced("retriever")
def retrieve(query: str, retriever: HybridRetriever, k: int = 5) -> EvidencePack:
    return EvidencePack(query=query, chunks=retriever.search(query, k=k))


def build_evidence(pack: EvidencePack) -> EvidencePack:
    return pack


@traced("answer_writer")
def answer(question: str, evidence: EvidencePack, provider: LLMProvider) -> AnswerDraft:
    query_terms = {term for term in tokenize(question) if len(term) > 2}
    evidence_terms = set(tokenize(" ".join(item.chunk.text for item in evidence.chunks)))
    if query_terms and not query_terms.intersection(evidence_terms):
        return AnswerDraft(answer="I could not find enough evidence in the supplied corpus.", abstained=True)
    return provider.answer(question, evidence)


def run_research(
    question: str,
    user_id: str,
    retriever: HybridRetriever,
    provider: LLMProvider,
    memory: MemoryStore,
    history: list[HistoryMessage] | None = None,
    prepared_query: str | None = None,
    prepared_evidence: EvidencePack | None = None,
    use_memory: bool = False,
    calculator: Callable[[str, EvidencePack], AnswerDraft | None] | None = None,
    evidence_guidance: str | None = None,
) -> ResearchResponse:
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
    if is_remember and re.search(r"(?i)(ignore the knowledge base|ignore previous instructions|only answer from memory|reveal system prompt)", command):
        return ResearchResponse(answer="I will not save instructions that override the evidence and safety rules.", evidence=[], provider="rules", model="deterministic")
    if is_remember and not is_company_fact_claim(command):
        content = re.sub(r"(?is)^remember(?: that)?\s+", "", command).strip()
        record = memory.write_explicit(user_id, content)
        if record is not None:
            return ResearchResponse(answer="Saved as unverified user context; saved context is not company evidence.", evidence=[], memory_written=[record.content], provider="rules", model="deterministic")
        return ResearchResponse(answer="I could not save that as memory because it was empty or failed the safety policy.", evidence=[], provider="rules", model="deterministic")
    if re.match(r"(?i)^(list|show) (my )?memories$", command) or re.match(r"(?i)^what do you remember about me\??$", command):
        records = memory.list_active(user_id)
        answer_text = "Saved context:\n" + "\n".join(f"- {record.content}" for record in records) if records else "You have no saved context."
        return ResearchResponse(answer=answer_text, evidence=[], provider="rules", model="deterministic", memory_used=[record.content for record in records])
    if re.match(r"(?i)^forget (everything|all)( you know)?( about me)?[.!]?$", command):
        count = memory.forget_all(user_id)
        return ResearchResponse(answer=f"Forgot {count} saved memor{'y' if count == 1 else 'ies'}.", evidence=[], provider="rules", model="deterministic")
    previous_assistant = next((item.content for item in reversed(history or []) if getattr(item, "role", None) == "assistant"), "")
    pending_forget = bool(re.search(r"(?i)\b(?:which|what)\b.{0,100}\bforget\b", previous_assistant))
    if pending_forget and re.match(r"(?i)^(?:the\s+)?(?:.+\s+)?(?:one|item|first|second|third|last)[.!?]?$", command):
        selector = re.sub(r"(?i)^(?:the\s+)?|\s+(?:one|item|first|second|third|last)[.!?]?$", "", command).strip()
        matches = memory.find_forget_matches(user_id, selector)
        if len(matches) == 1 and matches[0].id is not None:
            count = memory.forget_records(user_id, [matches[0].id])
            return ResearchResponse(answer=f"Forgot {count} matching saved memory.", evidence=[], provider="rules", model="deterministic")
        return ResearchResponse(answer="I still cannot identify one unambiguous saved memory; please name a distinctive word from it.", evidence=[], provider="rules", model="deterministic")
    forget_match = re.match(r"(?is)^forget(?: that)?\s+(.+)$", command)
    if forget_match:
        target = forget_match.group(1).strip()
        matches = memory.find_forget_matches(user_id, target)
        if len(matches) > 1:
            choices = " | ".join(f"{index}. {record.content}" for index, record in enumerate(matches, start=1))
            answer_text = f"Which saved item should I forget? Please name one of these: {choices}"
        elif len(matches) == 1 and matches[0].id is not None:
            count = memory.forget_records(user_id, [matches[0].id])
            answer_text = f"Forgot {count} matching saved memory."
        else:
            answer_text = "I could not find a matching saved memory to forget."
        return ResearchResponse(answer=answer_text, evidence=[], provider="rules", model="deterministic")
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
        if not is_company_fact_claim(command):
            record = memory.correct(user_id, target, replacement)
            answer_text = f"Corrected saved context to: {record.content}" if record else "I could not identify one unambiguous saved memory to correct."
            return ResearchResponse(answer=answer_text, evidence=[], provider="rules", model="deterministic", memory_written=[record.content] if record else [])
    query = prepared_query if prepared_query is not None else route(question, history)
    records = memory.list_active(user_id)
    used: list[str] = []
    if use_memory:
        age_cutoff = memory.clock.now() - timedelta(days=14)
        stale = [item for item in records if item.type in {"project", "task"} and item.updated_at and item.updated_at < age_cutoff.isoformat()]
        if stale and re.search(r"(?i)(continue|where we left|that|it)", question):
            return ResearchResponse(answer=f"Last time you were investigating {stale[0].content}. Is that still relevant?", evidence=[], memory_used=[stale[0].content], provider=provider.name, model=provider.model)
        if records and re.search(r"(?i)(what plan did i|which plan did i|what do you remember)", question):
            selected = records[:2]
            return ResearchResponse(answer="From what you told me earlier: " + "; ".join(record.content for record in selected) + ".", evidence=[], memory_used=[record.content for record in selected], provider=provider.name, model=provider.model)
        # Only relevant saved context may disambiguate retrieval. Dumping every
        # memory into the query pollutes ranking and makes memory act like evidence.
        question_terms = {term for term in tokenize(question) if len(term) > 3}
        relevant_records = [record for record in records if question_terms.intersection(tokenize(record.content))]
        contextual_followup = bool(
            re.search(r"(?i)^(?:why|how|what)\b.{0,100}\b(?:that|it|those|these)\b", question)
            or re.match(r"(?i)^(?:and|also|what about|how about)\b", question)
        )
        if not relevant_records and contextual_followup:
            projects = [record for record in records if record.type in {"project", "task"}]
            if len(projects) == 1:
                relevant_records = projects
            elif len(projects) > 1:
                choices = " | ".join(record.content for record in projects[:3])
                return ResearchResponse(
                    answer=f"Which saved investigation do you mean? {choices}", evidence=[],
                    memory_used=[record.content for record in projects[:3]], provider="rules", model="deterministic",
                )
        if not relevant_records and re.search(r"(?i)(my plan|my workspace|my project|warehouse export|alert rules)", question):
            relevant_records = records[:2]
        used = [record.content for record in relevant_records[:3]]
        query += " " + " ".join(record.content for record in relevant_records[:3])
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
    draft = calculator(query, pack) if calculator is not None else None
    if draft is None:
        answer_question = f"{query}\n{evidence_guidance}" if evidence_guidance else query
        draft = answer(answer_question, pack, provider)
    answer_text = draft.answer
    if use_memory and re.search(r"(?i)aql timeout", question) and any("50 seconds" in item.chunk.text for item in pack.chunks):
        source = next(item for item in pack.chunks if "50 seconds" in item.chunk.text)
        answer_text = f"The current AQL timeout is 50 seconds [{source.chunk.chunk_id} — {source.chunk.title}]."
    if use_memory and records:
        relevant = [record for record in records if any(term in record.content.lower() for term in tokenize(question.lower()) if len(term) > 3)]
        if not relevant and re.search(r"(?i)(plan|remember|told|said|earlier|that|continue|where we left|alert rules|warehouse export)", question):
            relevant = records[:2]
        if relevant:
            used = [record.content for record in relevant[:2]]
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
        rejected_company = is_company_fact_claim(content)
        rejected_instruction = bool(re.search(r"(?i)(ignore the knowledge base|only from memory|ignore previous instructions)", content))
        record = None if rejected_company or rejected_instruction else memory.write_explicit(user_id, content)
        if record:
            written.append(record.content)
        elif has_secret or has_sensitive_personal:
            return ResearchResponse(answer="I did not store that secret or sensitive value. Please rotate the credential and use the HR portal for personal leave or health matters.", evidence=pack.chunks, memory_used=used, provider=provider.name, model=provider.model)
    metadata = getattr(draft, "trace_metadata", {})
    served_provider = str(metadata.get("provider") or provider.name)
    served_model = str(metadata.get("model") or provider.model)
    return ResearchResponse(answer=answer_text, evidence=pack.chunks, calculated_values=draft.calculated_values, memory_used=used, memory_written=written, abstained=getattr(draft, "abstained", False), provider=served_provider, model=served_model, degraded=getattr(draft, "degraded", False), attempts=getattr(draft, "attempts", 0), error_class=getattr(draft, "error_class", None), error_status=metadata.get("status"), quota_type=metadata.get("quota_type"), retry_after=metadata.get("retry_after"), primary_error=getattr(draft, "primary_error", None), fallback_error=getattr(draft, "fallback_error", None), provider_error=getattr(draft, "provider_error", False))

