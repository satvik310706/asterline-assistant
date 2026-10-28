"""The selectable improved research graph.

The baseline remains in :mod:`research`; this graph adds explicit, inspectable
agent boundaries without moving policy decisions into prompts.
"""

from __future__ import annotations

import re
import hashlib
import time
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .llm import LLMProvider
from .memory import MemoryStore
from .models import EvidencePack, HistoryMessage, MemoryRecord, ResearchResponse, RetrievedChunk
from .calculator import calculate
from .research import build_evidence, is_company_fact_claim, retrieve, route, run_research
from .retrieval import HybridRetriever, tokenize
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
    numeric_valid: bool = True
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
    repair_count: int


_GENERIC_TOPIC_WORDS = {
    "after", "before", "current", "earlier", "effective", "from", "later", "month", "published",
    "records", "replaced", "replaces", "replacing", "superseded", "supersedes", "the", "this", "version",
}


def _topic_terms(text: str) -> set[str]:
    terms = {term for term in re.findall(r"[a-z][a-z0-9-]+", text.casefold()) if len(term) >= 3 and term not in _GENERIC_TOPIC_WORDS}
    # Treat compound forms such as cold-storage and on-call as their constituent
    # topic words too, so a supersession statement can match differently hyphenated text.
    terms.update(part for term in tuple(terms) for part in term.split("-") if len(part) >= 3 and part not in _GENERIC_TOPIC_WORDS)
    return terms


def _revision_order(item: Any) -> tuple[int, tuple[int, ...]]:
    published = re.sub(r"\D", "", str(item.chunk.published))
    version = tuple(int(part) for part in re.findall(r"\d+", str(item.chunk.version)))
    return int(published or 0), version


def _split_sentences(text: str) -> list[str]:
    # Decimal points stay within a sentence (99.92, 2.3.1).
    return [part.strip() for part in re.split(r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z])", text) if part.strip()]


def _scope_values(text: str) -> dict[str, set[str]]:
    return {
        "plan": {value.casefold() for value in re.findall(r"\b(launch|team|enterprise)\b", text, re.IGNORECASE)},
        "region": {value.casefold().replace(" ", "-") for value in re.findall(r"\b(EU|US|Asia[- ]Pacific)\b", text, re.IGNORECASE)},
    }


def _apply_scope(question: str, pack: EvidencePack, labels: dict[str, str]) -> dict[str, str]:
    requested_scope = _scope_values(question)
    for item in pack.chunks:
        chunk_id = item.chunk.chunk_id
        if labels.get(chunk_id) != "CURRENT":
            continue
        stated = _scope_values(item.chunk.text)
        for dimension, requested in requested_scope.items():
            if requested and stated[dimension] and not requested.intersection(stated[dimension]):
                labels[chunk_id] = "NOT_APPLICABLE"
                break
    return labels


def _label_supersession(question: str, pack: EvidencePack) -> dict[str, str]:
    """Mark an older chunk only when newer, same-topic evidence names its value as replaced."""
    labels = {item.chunk.chunk_id: "CURRENT" for item in pack.chunks}
    requested_version = re.search(r"\b(?:version|in|at)\s+(\d+(?:\.\d+){1,2})\b", question, re.IGNORECASE)
    if requested_version:
        requested = requested_version.group(1)
        found = False
        for item in pack.chunks:
            matches = str(item.chunk.version) == requested or str(item.chunk.version).startswith(requested + ".")
            labels[item.chunk.chunk_id] = "CURRENT" if matches else "NOT_APPLICABLE"
            found |= matches
        if not found:
            labels = {item.chunk.chunk_id: "NOT_APPLICABLE" for item in pack.chunks}
        return _apply_scope(question, pack, labels)

    markers = re.compile(r"\b(?:supersedes|superseded|replaces|replaced|replacing|earlier versions used)\b", re.IGNORECASE)
    for newer in pack.chunks:
        supersession_sentences = [
            sentence for sentence in _split_sentences(newer.chunk.text)
            if markers.search(sentence) and re.search(r"\d", sentence)
        ]
        if not supersession_sentences:
            continue
        for older in pack.chunks:
            if older.chunk.chunk_id == newer.chunk.chunk_id or _revision_order(newer) <= _revision_order(older):
                continue
            older_values = set(re.findall(r"\d+(?:\.\d+)?", older.chunk.text))
            explicit_topic_conflict = any(
                bool(_topic_terms(sentence) & _topic_terms(older.chunk.text))
                and bool(set(re.findall(r"\d+(?:\.\d+)?", sentence)) & older_values)
                for sentence in supersession_sentences
            )
            if explicit_topic_conflict:
                labels[older.chunk.chunk_id] = "SUPERSEDED"
    return _apply_scope(question, pack, labels)


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
    previous_assistant = next((item.content for item in reversed(history) if item.role == "assistant"), "")
    if re.search(r"(?i)\b(?:which|what)\b.{0,100}\bforget\b", previous_assistant) and re.match(r"^(?:the\s+)?(?:.+\s+)?(?:one|item|first|second|third|last)[.!?]?$", text):
        return "forget"
    if is_chitchat_question(text):
        return "chitchat"
    # Only treat as follow-up if referencing previous context (pronouns) or starting with a conjunction.
    if history and (re.search(r"\b(it|that|those|these|them|the former|the latter)\b", text) or re.match(r"^(?:and|also|what about|how about)\b", text)):
        return "followup"
    return "question"


def is_chitchat_question(question: str) -> bool:
    text = question.strip().lower()
    return bool(re.fullmatch(r"(?:hi|hello|hey|thanks|thank you)[.! ]*", text)) or is_assistant_or_corpus_meta_question(text)


def is_assistant_or_corpus_meta_question(question: str) -> bool:
    """Identify questions answerable from app identity and loaded corpus inventory."""
    role = re.search(r"\b(who are you|your role|what can you do|your purpose|what do you do|rules|instructions|training data)\b", question, re.IGNORECASE)
    inventory = re.search(
        r"\bhow many\s+(?:knowledge[- ]base\s+)?(?:chunks|records|documents|docs)\b"
        r"(?:(?!\b(?:support|mention|cover|cite)\b).){0,90}\b"
        r"(?:do you have|are there|are in|does (?:the )?(?:corpus|knowledge base) have)\b"
        r"|\bhow many\s+(?:chunks|records|documents|docs)\b.{0,60}\b(?:corpus|knowledge base)\b",
        question,
        re.IGNORECASE,
    )
    return bool(role or inventory)


@traced("router")
def router_agent(state: GraphState) -> GraphState:
    intent = _intent(state["question"], state.get("history", []))
    query = route(state["question"], state.get("history", [])) if intent == "followup" else state["question"].strip()
    result = RouterResult(intent=intent, standalone_query=query, sub_questions=[query])
    return {"router": result}


@traced("memory_reader")
def memory_reader_agent(state: GraphState) -> GraphState:
    records = state["memory"].list_active(state["user_id"])
    return {"memory_reader": MemoryReaderResult(records=records)}


@traced("retriever")
def retriever_agent(state: GraphState) -> GraphState:
    query = state["router"].standalone_query
    # Memory is read as unverified context by the answer path; it never enters
    # the evidence pack or indiscriminately pollutes the retrieval query.
    pack = build_evidence(retrieve(query, state["retriever"]))
    requested_plans = {plan.casefold() for plan in re.findall(r"\b(Launch|Team|Enterprise)\b", state["question"], re.IGNORECASE)}
    references_plan_retention = [
        item for item in pack.chunks
        if re.search(r"plan[- ]specific searchable retention", item.chunk.text, re.IGNORECASE)
    ]
    if requested_plans and references_plan_retention:
        # Follow an explicit corpus reference to plan-specific searchable-retention
        # facts, even when hybrid top-k retrieved the policy but missed the plan table.
        companions = [
            chunk for chunk in state["retriever"].chunks
            if re.search(r"searchable run records|searchable retention", chunk.text, re.IGNORECASE)
            and requested_plans.intersection({plan.casefold() for plan in re.findall(r"\b(Launch|Team|Enterprise)\b", chunk.text, re.IGNORECASE)})
        ]
        selected = {item.chunk.chunk_id: item for item in pack.chunks}
        required_ids = {item.chunk.chunk_id for item in references_plan_retention}
        for chunk in companions:
            required_ids.add(chunk.chunk_id)
            selected.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
        ordered_required = [selected[chunk_id] for chunk_id in selected if chunk_id in required_ids]
        ordered_other = sorted(
            (item for chunk_id, item in selected.items() if chunk_id not in required_ids),
            key=lambda item: (-item.rrf_score, item.dense_rank or 999, item.chunk.chunk_id),
        )
        pack = EvidencePack(query=pack.query, chunks=(ordered_required + ordered_other)[:5])
    return {"evidence": pack}


@traced("evidence_analyst")
def evidence_analyst_agent(state: GraphState) -> GraphState:
    pack = state["evidence"]
    labels = _label_supersession(state["question"], pack)
    sufficient = "answerable" if any(label == "CURRENT" for label in labels.values()) else "unanswerable"
    return {"assessment": EvidenceAssessment(evidence=pack, labels=labels, sufficiency=sufficient)}


def _supersession_answer(state: GraphState) -> ResearchResponse | None:
    """Quote only retrieved, scope-matched source text when explicit supersession resolves a conflict."""
    assessment = state["assessment"]
    labels = assessment.labels
    old_items = [item for item in assessment.evidence.chunks if labels.get(item.chunk.chunk_id) == "SUPERSEDED"]
    if not old_items:
        return None

    question_sentences = _split_sentences(state["question"])
    focus = state["question"]
    # In a context-setting sentence followed by a fresh wh-question, resolve the
    # requested topic rather than the preceding mention. Pronoun follow-ups keep
    # the full text so their antecedent remains available.
    if len(question_sentences) > 1 and re.match(r"(?i)^(?:which|what|how|who|when|where)\b", question_sentences[-1]):
        focus = question_sentences[-1]
    question_terms = _topic_terms(focus)
    question_numbers = set(re.findall(r"\d+(?:\.\d+)?", focus))
    marker = re.compile(r"\b(?:supersedes|superseded|replaces|replaced|replacing|earlier versions used)\b", re.IGNORECASE)
    def related_sentence(sentence: str) -> bool:
        shared_terms = question_terms & _topic_terms(sentence)
        shared_numbers = question_numbers & set(re.findall(r"\d+(?:\.\d+)?", sentence))
        return (len(shared_terms) >= 2 or (bool(shared_terms) and bool(shared_numbers))) and bool(re.search(r"\d", sentence))

    candidates: list[tuple[Any, list[Any]]] = []
    for current in assessment.evidence.chunks:
        if labels.get(current.chunk.chunk_id) != "CURRENT" or not marker.search(current.chunk.text):
            continue
        current_terms = _topic_terms(current.chunk.text)
        related_old = []
        for old in old_items:
            shared_terms = question_terms & current_terms & _topic_terms(old.chunk.text)
            shared_numbers = question_numbers & set(re.findall(r"\d+(?:\.\d+)?", current.chunk.text)) & set(re.findall(r"\d+(?:\.\d+)?", old.chunk.text))
            if len(shared_terms) >= 2 or (shared_terms and shared_numbers):
                related_old.append(old)
        if not related_old:
            continue
        relevant_sentences = [sentence for sentence in _split_sentences(current.chunk.text) if related_sentence(sentence)]
        if relevant_sentences:
            candidates.append((current, related_old))
    if not candidates:
        return None

    current, related_old = candidates[0]
    current_sentences = [
        sentence for sentence in _split_sentences(current.chunk.text)
        if related_sentence(sentence)
    ]
    current_citation = f"[{current.chunk.chunk_id} — {current.chunk.title}]"
    answer_parts = ["Current source: " + " ".join(current_sentences) + " " + current_citation]
    seen_old: set[str] = set()
    for old in related_old:
        old_sentences = [
            sentence for sentence in _split_sentences(old.chunk.text)
            if related_sentence(sentence)
        ]
        if not old_sentences or old.chunk.chunk_id in seen_old:
            continue
        seen_old.add(old.chunk.chunk_id)
        old_citation = f"[{old.chunk.chunk_id} — {old.chunk.title}]"
        answer_parts.append(
            "The older source is superseded: " + " ".join(old_sentences) + " " + old_citation
        )
    return ResearchResponse(
        answer=" ".join(answer_parts),
        evidence=assessment.evidence.chunks,
        provider="rules",
        model="deterministic",
    )


def _delegated_authority_answer(state: GraphState) -> ResearchResponse | None:
    """Follow an explicit corpus statement naming the document that owns a topic."""
    question = state["question"]
    if not re.search(r"\b(which|what)\s+(?:document|source|runbook)\b", question, re.IGNORECASE) or not re.search(r"\b(define|own|authoritative|deadline|procedure)\w*\b", question, re.IGNORECASE):
        return None
    evidence = state["assessment"].evidence
    for authority in evidence.chunks:
        ownership = re.search(
            r"\b(?:definitions|deadlines|procedures?)\s+remain\s+owned\s+by\s+the\s+([^.;]+)",
            authority.chunk.text,
            re.IGNORECASE,
        )
        if not ownership:
            continue
        owner_name = ownership.group(1).strip()
        requested_identifiers = {item.casefold() for item in re.findall(r"\b[A-Z][A-Z0-9]+-\d+\b", question)}
        source = next((
            item for item in evidence.chunks
            if owner_name.casefold() in item.chunk.title.casefold()
            and requested_identifiers.issubset({identifier.casefold() for identifier in re.findall(r"\b[A-Z][A-Z0-9]+-\d+\b", item.chunk.text)})
        ), None)
        if source is None:
            continue
        source_sentences = _split_sentences(source.chunk.text)
        answer_sentences = [
            sentence for sentence in source_sentences
            if requested_identifiers.intersection(item.casefold() for item in re.findall(r"\b[A-Z][A-Z0-9]+-\d+\b", sentence))
            and re.search(r"\d", sentence)
        ]
        if requested_identifiers and not requested_identifiers.issubset(
            {identifier.casefold() for sentence in answer_sentences for identifier in re.findall(r"\b[A-Z][A-Z0-9]+-\d+\b", sentence)}
        ):
            continue
        if not answer_sentences:
            continue
        answer = (
            f"The {owner_name} defines these deadlines: {' '.join(answer_sentences)} "
            f"[{source.chunk.chunk_id} — {source.chunk.title}] "
            f"The documentation confirms that these definitions remain owned by the {owner_name}. "
            f"[{authority.chunk.chunk_id} — {authority.chunk.title}]"
        )
        return ResearchResponse(answer=answer, evidence=evidence.chunks, provider="rules", model="deterministic")
    return None


def _incident_answer(state: GraphState) -> ResearchResponse | None:
    """Answer incident-ID questions directly from the matching retrieved records."""
    incident_ids = set(re.findall(r"\bINC-\d{4}-\d{2}\b", state["question"], re.IGNORECASE))
    if len(incident_ids) != 1:
        return None
    incident_id = next(iter(incident_ids)).upper()
    evidence = state["assessment"].evidence
    matching = [
        item for item in evidence.chunks
        if incident_id in f"{item.chunk.title} {item.chunk.text}".upper()
    ]
    if not matching:
        return None
    matching.sort(key=lambda item: item.chunk.chunk_id)
    answer = " ".join(
        f"{item.chunk.text.strip()} [{item.chunk.chunk_id} — {item.chunk.title}]"
        for item in matching
    )
    return ResearchResponse(answer=answer, evidence=evidence.chunks, provider="rules", model="deterministic")


@traced("answer_writer")
def answer_writer_agent(state: GraphState) -> GraphState:
    intent = state["router"].intent
    if state["router"].intent == "chitchat":
        question = state["question"].lower()
        parts: list[str] = []
        if re.search(r"\b(who are you|your role|what can you do|your purpose|what do you do|rules|instructions|training data)\b", question):
            parts.append("I'm Asterline's research assistant. I answer company questions strictly from the supplied knowledge base, cite supporting sources ([chunk_id — title]), maintain strict user memory isolation, and keep saved user context separate from corporate evidence.")
        if re.search(r"\bhow many\s+(?:knowledge[- ]base\s+)?(?:chunks|records|documents|docs)\b|\b(?:corpus|knowledge base)\s+(?:has|contains)\s+how many\b", question):
            chunks = state["retriever"].chunks
            unique_docs = {chunk.doc_id for chunk in chunks}
            parts.append(f"The loaded knowledge base contains {len(chunks)} chunks across {len(unique_docs)} documents.")
        if not parts:
            parts.append("Hi! I can help with Asterline's plans, APIs, incidents, and policies. What would you like to know?")
        return {"response": ResearchResponse(
            answer=" ".join(parts),
            evidence=[], provider="rules", model="deterministic",
        )}
    if intent in {"forget", "list"} or (intent == "remember" and not is_company_fact_claim(state["question"])) or (intent == "correct" and not is_company_fact_claim(state["question"])):
        command_query = state["router"].standalone_query
        response = run_research(
            state["question"], state["user_id"], state["retriever"], state["provider"], state["memory"],
            history=state.get("history", []), prepared_query=command_query,
            prepared_evidence=EvidencePack(query=command_query, chunks=[]), use_memory=True,
        )
        return {"response": response}
    if intent in {"question", "followup"}:
        incident_response = _incident_answer(state)
        if incident_response is not None:
            return {"response": incident_response}
        authority_response = _delegated_authority_answer(state)
        if authority_response is not None:
            return {"response": authority_response}
        supersession_response = _supersession_answer(state)
        if supersession_response is not None:
            return {"response": supersession_response}
    records = state["memory_reader"].records
    query_terms = {term for term in tokenize(state["question"]) if len(term) > 3}
    relevant_memory = any(query_terms.intersection(tokenize(record.content)) for record in records)
    memory_request = bool(re.search(r"(?i)(remember|plan|investigat|where we left|what do you remember)", state["question"]))
    assessment = state["assessment"]
    labels = assessment.labels
    ordered_chunks = sorted(
        assessment.evidence.chunks,
        key=lambda item: (labels.get(item.chunk.chunk_id) != "CURRENT", item.dense_rank or 0, -item.rrf_score),
    )
    answer_evidence = EvidencePack(query=assessment.evidence.query, chunks=ordered_chunks)
    evidence_guidance = "Use CURRENT chunks for current facts. A SUPERSEDED chunk is historical only; if you mention its value, explicitly state that it was superseded. Do not treat NOT_APPLICABLE chunks as evidence for the requested scope."
    if state.get("repair_count", 0):
        reason = state.get("verifier").reason if state.get("verifier") else "the prior draft did not pass validation"
        evidence_guidance += f" Repair the previous draft after this verifier failure: {reason}. Remove unsupported claims and cite exact retrieved IDs and titles."
    if assessment.sufficiency == "unanswerable":
        return {"response": ResearchResponse(
            answer="I could not find current evidence for the requested scope in the supplied corpus.",
            evidence=answer_evidence.chunks,
            abstained=True,
            provider="rules",
            model="deterministic",
        )}
    response = run_research(
        state["question"], state["user_id"], state["retriever"], state["provider"], state["memory"],
        history=state.get("history", []), prepared_query=state["router"].standalone_query,
        prepared_evidence=answer_evidence, use_memory=relevant_memory or memory_request,
        calculator=calculate,
        evidence_guidance=evidence_guidance,
    )
    return {"response": response}


def _normalized_numbers(text: str) -> set[str]:
    numbers: set[str] = set()
    for token in re.findall(r"(?<![A-Za-z])\$?\d[\d,]*(?:\.\d+)?%?", text):
        try:
            value = Decimal(token.replace("$", "").replace("%", "").replace(",", ""))
        except InvalidOperation:
            continue
        numbers.add(format(value.normalize(), "f"))
    return numbers


@traced("verifier")
def verifier_agent(state: GraphState) -> GraphState:
    response = state["response"]
    # User-memory actions are policy operations, not company-fact claims.
    # Their numeric counts (for example, "Forgot 1 memory") are not evidence.
    if state["router"].intent in {"forget", "list", "remember", "correct"} and response.provider == "rules":
        return {"verifier": VerifierResult(valid=True, cited_ids=[], numeric_valid=True)}
    known = {item.chunk.chunk_id: item for item in state["evidence"].chunks}
    pairs = re.findall(r"\[([^\s—]+)\s+—\s+([^\]]+)\]", response.answer)
    cited = [chunk_id for chunk_id, _ in pairs]
    citations_valid = all(chunk_id in known and title.strip() == known[chunk_id].chunk.title for chunk_id, title in pairs)
    cited_sources = " ".join(known[chunk_id].chunk.text for chunk_id in cited if chunk_id in known)
    answer_body = re.sub(r"\[[^\]]+\]", "", response.answer)
    answer_body = answer_body.split("\nFrom what you told me earlier:", 1)[0]
    answer_numbers = _normalized_numbers(answer_body)
    evidence_numbers = _normalized_numbers(cited_sources)
    calculator_numbers = {number for value in response.calculated_values for number in _normalized_numbers(value)}
    numeric_valid = answer_numbers <= evidence_numbers | calculator_numbers
    valid = citations_valid and numeric_valid
    reasons = []
    if not citations_valid:
        reasons.append("citation id/title outside evidence pack")
    if not numeric_valid:
        reasons.append("numeric claim absent from cited evidence and calculator output")
    result = VerifierResult(valid=valid, cited_ids=cited, numeric_valid=numeric_valid, reason="; ".join(reasons) or None)
    if not valid:
        repair_count = state.get("repair_count", 0) + 1
        if repair_count > 1:
            reason = result.reason or "the answer did not pass grounding checks"
            response.answer = f"I could not verify the answer after one repair, so I will not guess. Verification issue: {reason}."
            response.abstained = True
            response.error_class = "VerifierFailure"
        return {"response": response, "verifier": result, "repair_count": repair_count}
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
    graph.add_conditional_edges("router", lambda state: "chitchat" if state["router"].intent == "chitchat" else "research", {"chitchat": "answer_writer", "research": "memory_reader"})
    graph.add_edge("memory_reader", "retriever")
    graph.add_edge("retriever", "evidence_analyst")
    graph.add_edge("evidence_analyst", "answer_writer")
    graph.add_conditional_edges("answer_writer", lambda state: "done" if state["router"].intent == "chitchat" else "verify", {"done": END, "verify": "verifier"})
    graph.add_conditional_edges(
        "verifier",
        lambda state: "repair" if not state["verifier"].valid and state.get("repair_count", 0) == 1 else "done",
        {"repair": "answer_writer", "done": "memory_writer"},
    )
    graph.add_edge("memory_writer", END)
    return graph.compile()


def run_improved_research(question: str, user_id: str, retriever: HybridRetriever, provider: LLMProvider, memory: MemoryStore, history: list[HistoryMessage] | None = None, on_step: Callable[[str, float], None] | None = None) -> ResearchResponse:
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
        initial: GraphState = {"question": question, "user_id": user_id, "history": history or [], "retriever": retriever, "provider": provider, "memory": memory}
        state: dict[str, Any] = dict(initial)
        last_event = time.perf_counter()
        for update in _build_graph().stream(initial, stream_mode="updates"):
            now = time.perf_counter()
            for node_name, node_update in update.items():
                state.update(node_update)
                if on_step is not None:
                    on_step(node_name, now - last_event)
                last_event = now
        return state["response"]
    finally:
        clear_trace_context()
