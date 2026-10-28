"""LLM providers with bounded retries, circuit breaking, and clean fallback."""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.request
from collections.abc import Callable
from typing import Any

from google import genai

from .models import AnswerDraft, EvidencePack

TRANSIENT_CODES = {429, 500, 503, 504}
CONFIG_CODES = {400, 401, 403, 404}
CONFIG_MARKERS = ("model not found", "model no longer available", "not_found", "not found")
DECLINE_MARKERS = re.compile(r"(?i)(unable to answer|cannot answer|not (?:stated|published|available|mentioned)|no information|cannot confirm|evidence does not|not in the (?:provided )?(?:documents|evidence))")


class ProviderFailure(RuntimeError):
    def __init__(self, message: str, *, error_class: str, transient: bool, status_code: int | None = None, retry_after: float | None = None, attempts: int = 0, detail: str | None = None, raw_reply: str | None = None, quality: bool = False) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.transient = transient
        self.status_code = status_code
        self.retry_after = retry_after
        self.attempts = attempts
        self.detail = detail or message
        self.raw_reply = raw_reply
        self.quality = quality


class CircuitBreaker:
    def __init__(self, threshold: int = 3, cooldown_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self.consecutive_failures = 0
        self.opened_at: float | None = None

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if self.clock() - self.opened_at >= self.cooldown_seconds:
            self.opened_at = None
            self.consecutive_failures = 0
            return False
        return True

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.opened_at = None

    def record_transient_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.threshold:
            self.opened_at = self.clock()


def _status_code(error: Exception) -> int | None:
    for name in ("code", "status_code", "status"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    match = re.search(r"\b(400|401|403|404|429|500|503|504)\b", str(error))
    return int(match.group(1)) if match else None


def _retry_after(error: Exception) -> float | None:
    value = getattr(error, "retry_after", None)
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
    headers = getattr(error, "headers", None)
    if headers is None and getattr(error, "response", None) is not None:
        headers = getattr(error.response, "headers", None)
    if headers:
        raw = headers.get("Retry-After") or headers.get("retry-after")
        try:
            return max(0.0, float(raw)) if raw is not None else None
        except (TypeError, ValueError):
            return None
    return None


def classify_failure(error: Exception, attempts: int = 0) -> ProviderFailure:
    code = _status_code(error)
    detail = _sanitize(str(error)) or "provider request failed"
    config_error = code in CONFIG_CODES or any(marker in detail.lower() for marker in CONFIG_MARKERS)
    transient = not config_error and (code in TRANSIENT_CODES or isinstance(error, TimeoutError) or "timeout" in error.__class__.__name__.lower())
    message = "Gemini rejected the model/key: check GEMINI_MODEL and your key." if config_error else "provider request failed"
    return ProviderFailure(message, error_class=error.__class__.__name__, transient=transient, status_code=code, retry_after=_retry_after(error), attempts=attempts, detail=detail)


def _sanitize(value: str) -> str:
    return re.sub(r"(?i)(api[_ -]?key|token|password|authorization)\s*[:=]?\s*[^\s,;]+", r"\1=<redacted>", value).replace("\n", " ")[:240]


def format_failure(provider: str, model: str, failure: ProviderFailure, role: str) -> str:
    status = f" {failure.status_code}" if failure.status_code is not None else ""
    raw = f" raw reply (first 300): {_sanitize(failure.raw_reply)}" if failure.raw_reply is not None else ""
    return f"{role}: {provider}/{model} failed: {failure.error_class}{status} {_sanitize(failure.detail)}{raw}"


def parse_text_answer(text: str, evidence: EvidencePack, *, attempt: int, degraded: bool) -> AnswerDraft:
    cleaned = re.sub(r"^\s*```(?:json|text)?\s*|\s*```\s*$", "", text or "", flags=re.IGNORECASE | re.DOTALL).strip()
    if not cleaned:
        raise ProviderFailure("model returned an empty reply", error_class="EmptyResponse", transient=False, attempts=attempt, raw_reply=(text or "")[:300])
    citations = re.findall(r"\[([^\s—]+)\s+—\s+([^\]]+)\]", cleaned)
    known = {item.chunk.chunk_id for item in evidence.chunks}
    if any(chunk_id not in known for chunk_id, _ in citations):
        raise ProviderFailure("model cited a chunk outside the evidence pack", error_class="InvalidCitation", transient=False, attempts=attempt, raw_reply=cleaned[:300], quality=True)
    if evidence.chunks and not citations:
        if not DECLINE_MARKERS.search(cleaned):
            raise ProviderFailure("factual reply did not include a citation", error_class="MissingCitation", transient=False, attempts=attempt, raw_reply=cleaned[:300], quality=True)
        return AnswerDraft(answer=cleaned, cited_chunk_ids=[], abstained=True, degraded=degraded, attempts=attempt, trace_metadata={"provider": "ollama" if degraded else "gemini", "attempts": attempt, "degraded": degraded, "error_class": None})
    return AnswerDraft(answer=cleaned, cited_chunk_ids=[chunk_id for chunk_id, _ in citations], degraded=degraded, attempts=attempt, trace_metadata={"provider": "ollama" if degraded else "gemini", "attempts": attempt, "degraded": degraded, "error_class": None})


class LLMProvider:
    name = "unknown"
    model = "unknown"

    def __init__(self) -> None:
        self.calls = 0
        self.last_metadata: dict[str, str | int | bool | None] = {}

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        raise NotImplementedError

    def preflight(self, evidence: EvidencePack) -> None:
        self.answer("Reply briefly using the evidence.", evidence)


class FakeLLM(LLMProvider):
    name = "fake"
    model = "deterministic"

    def __init__(self) -> None:
        super().__init__()

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        self.calls += 1
        if not evidence.chunks:
            return AnswerDraft(answer="I could not find enough evidence in the supplied corpus.", abstained=True, attempts=1)
        citations = [f"[{item.chunk.chunk_id} — {item.chunk.title}]" for item in evidence.chunks[:2]]
        text = " ".join(item.chunk.text for item in evidence.chunks[:2])
        return AnswerDraft(answer=f"Based on the supplied corpus: {text} {' '.join(citations)}", cited_chunk_ids=[item.chunk.chunk_id for item in evidence.chunks[:2]], attempts=1, trace_metadata={"provider": self.name, "model": self.model, "attempts": 1, "degraded": False, "error_class": None})


class GeminiLLM(LLMProvider):
    name = "gemini"

    def __init__(self, model: str, *, breaker: CircuitBreaker | None = None, sleep_fn: Callable[[float], None] = time.sleep, random_fn: Callable[[], float] = random.random, max_attempts: int = 4, max_total_wait: float = 30.0) -> None:
        super().__init__()
        self.model = model
        self.breaker = breaker or CircuitBreaker()
        self.sleep_fn = sleep_fn
        self.random_fn = random_fn
        self.max_attempts = max_attempts
        self.max_total_wait = max_total_wait
        self.client: Any = None
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        if self.breaker.is_open():
            raise ProviderFailure("Gemini circuit breaker is open", error_class="CircuitOpen", transient=True)
        if self.client is None:
            raise ProviderFailure("Gemini is not configured", error_class="ConfigurationError", transient=False)
        context = "\n".join(f"{x.chunk.chunk_id} | {x.chunk.title} | {x.chunk.text}" for x in evidence.chunks)
        prompt = f"Answer only from this evidence in plain text. Cite each claim exactly as [chunk_id — title]. If insufficient, abstain.\nQuestion: {question}\nEvidence:\n{context}"
        waited = 0.0
        parse_retries = 0
        for attempt in range(1, self.max_attempts + 1):
            self.calls += 1
            try:
                instruction = prompt if parse_retries == 0 else f"{prompt}\nReturn only a short plain-text answer and valid citations; do not use JSON or Markdown fences."
                response = self.client.models.generate_content(model=self.model, contents=instruction, config={"temperature": 0})
                self.breaker.record_success()
                draft = parse_text_answer(response.text or "", evidence, attempt=attempt, degraded=False)
                draft.trace_metadata = {"provider": self.name, "model": self.model, "attempts": attempt, "degraded": False, "error_class": None}
                return draft
            except ProviderFailure as failure:
                if failure.error_class in {"InvalidCitation", "MissingCitation", "EmptyResponse"} and parse_retries == 0:
                    parse_retries += 1
                    continue
                raise failure from None
            except Exception as error:
                failure = classify_failure(error, attempts=attempt)
                if not failure.transient:
                    raise failure from None
                if attempt == self.max_attempts:
                    self.breaker.record_transient_failure()
                    failure.attempts = attempt
                    raise failure from None
                delay = failure.retry_after if failure.retry_after is not None else (2 ** (attempt - 1)) + self.random_fn()
                delay = min(delay, max(0.0, self.max_total_wait - waited))
                if delay <= 0:
                    self.breaker.record_transient_failure()
                    failure.attempts = attempt
                    raise failure from None
                self.sleep_fn(delay)
                waited += delay
        raise ProviderFailure("Gemini request failed", error_class="RetryExhausted", transient=True, attempts=self.max_attempts)

    def preflight(self, evidence: EvidencePack) -> None:
        if self.breaker.is_open():
            raise ProviderFailure("Gemini circuit breaker is open", error_class="CircuitOpen", transient=True)
        if self.client is None:
            raise ProviderFailure("Gemini is not configured", error_class="ConfigurationError", transient=False)
        context = "\n".join(f"{x.chunk.chunk_id} | {x.chunk.title} | {x.chunk.text}" for x in evidence.chunks)
        try:
            self.calls += 1
            response = self.client.models.generate_content(model=self.model, contents=f"Reply briefly using this evidence.\n{context}", config={"temperature": 0})
            if not (response.text or "").strip():
                raise ProviderFailure("model returned an empty reply", error_class="EmptyResponse", transient=False)
        except ProviderFailure:
            raise
        except Exception as error:
            raise classify_failure(error, attempts=1) from None


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, request_fn: Callable[[str], str] | None = None) -> None:
        super().__init__()
        self.model = model
        self.request_fn = request_fn or self._request

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        context = "\n".join(f"SOURCE {x.chunk.chunk_id} | {x.chunk.title}\n{x.chunk.text}" for x in evidence.chunks)
        example = f"Example citation using an actual source: [{evidence.chunks[0].chunk.chunk_id} — {evidence.chunks[0].chunk.title}]" if evidence.chunks else "If there is no evidence, say you cannot answer."
        errors: list[ProviderFailure] = []
        for attempt in (1, 2):
            self.calls += 1
            try:
                prompt = f"Answer briefly using only the evidence. Copy source ids and titles exactly from the evidence headers. {example}\nQuestion: {question}\nEvidence:\n{context}"
                if attempt == 2:
                    prompt = f"Answer in one or two sentences. Copy one exact citation from the evidence headers. {example} Q: {question}\n{context}"
                text = self.request_fn(prompt)
                return self._parse_text(text, evidence, attempt)
            except ProviderFailure as error:
                errors.append(error)
                if error.error_class in {"InvalidCitation", "MissingCitation", "ResponseParseError", "EmptyResponse"} and attempt == 1:
                    continue
                raise error from None
            except Exception as error:
                raise classify_failure(error, attempts=attempt) from None
        raise errors[-1] if errors else ProviderFailure("Ollama returned no answer", error_class="EmptyResponse", transient=False, attempts=2)

    def _request(self, prompt: str) -> str:
        payload = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        request = urllib.request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            try:
                result = json.loads(response.read())
            except json.JSONDecodeError as error:
                raise ProviderFailure("Ollama response was not valid JSON", error_class="ResponseParseError", transient=False, attempts=1, detail=str(error)) from None
        return str(result.get("response", ""))

    def _parse_text(self, text: str, evidence: EvidencePack, attempt: int) -> AnswerDraft:
        draft = parse_text_answer(text, evidence, attempt=attempt, degraded=True)
        draft.trace_metadata = {"provider": self.name, "model": self.model, "attempts": attempt, "degraded": True, "error_class": None}
        return draft

    def preflight(self, evidence: EvidencePack) -> None:
        try:
            text = self.request_fn(f"Reply briefly using this evidence.\n{chr(10).join(item.chunk.text for item in evidence.chunks)}")
            if not text.strip():
                raise ProviderFailure("Ollama returned an empty reply", error_class="EmptyResponse", transient=False)
        except ProviderFailure:
            raise
        except Exception as error:
            raise classify_failure(error, attempts=1) from None


class ResilientLLM(LLMProvider):
    """Use Gemini when available, then Ollama, never exposing provider exceptions."""

    def __init__(self, primary: GeminiLLM, fallback: OllamaLLM) -> None:
        super().__init__()
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.model = primary.model

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        try:
            draft = self.primary.answer(question, evidence)
            self.calls = self.primary.calls
            self.name, self.model = self.primary.name, self.primary.model
            return draft
        except ProviderFailure as primary_error:
            if primary_error.quality:
                return AnswerDraft(answer=primary_error.raw_reply or str(primary_error), error_class=primary_error.error_class, attempts=primary_error.attempts, trace_metadata={"provider": self.primary.name, "model": self.primary.model, "attempts": primary_error.attempts, "degraded": False, "error_class": primary_error.error_class})
            try:
                draft = self.fallback.answer(question, evidence)
                self.calls = self.primary.calls + self.fallback.calls
                self.name, self.model = self.fallback.name, self.fallback.model
                draft.degraded = True
                draft.attempts = primary_error.attempts + max(1, getattr(draft, "attempts", 1))
                primary_text = format_failure(self.primary.name, self.primary.model, primary_error, "primary")
                draft.primary_error = primary_text
                draft.error_class = primary_error.error_class
                draft.trace_metadata = {"provider": self.name, "model": self.model, "attempts": primary_error.attempts + 1, "degraded": True, "error_class": primary_error.error_class, "primary_error": primary_text, "fallback_error": None}
                return draft
            except ProviderFailure as fallback_error:
                self.calls = self.primary.calls + self.fallback.calls
                self.name, self.model = "unavailable", self.fallback.model
                message = "The language model is temporarily unavailable; here are the most relevant sources:"
                primary_text = format_failure(self.primary.name, self.primary.model, primary_error, "primary")
                fallback_text = format_failure(self.fallback.name, self.fallback.model, fallback_error, "fallback")
                return AnswerDraft(answer=message, degraded=True, provider_error=True, attempts=primary_error.attempts + fallback_error.attempts, error_class=primary_error.error_class, primary_error=primary_text, fallback_error=fallback_text, trace_metadata={"provider": "unavailable", "model": self.fallback.model, "attempts": primary_error.attempts + fallback_error.attempts, "degraded": True, "error_class": primary_error.error_class, "primary_error": primary_text, "fallback_error": fallback_text})

    def preflight(self, evidence: EvidencePack) -> None:
        try:
            self.primary.preflight(evidence)
        except ProviderFailure as primary_error:
            try:
                self.fallback.preflight(evidence)
            except ProviderFailure as fallback_error:
                primary_text = format_failure(self.primary.name, self.primary.model, primary_error, "primary")
                fallback_text = format_failure(self.fallback.name, self.fallback.model, fallback_error, "fallback")
                raise ProviderFailure(f"{primary_text}; {fallback_text}", error_class=primary_error.error_class, transient=False, status_code=primary_error.status_code, attempts=primary_error.attempts + fallback_error.attempts, detail=f"{primary_text}; {fallback_text}") from None
            raise ProviderFailure(format_failure(self.primary.name, self.primary.model, primary_error, "primary"), error_class=primary_error.error_class, transient=False, status_code=primary_error.status_code, attempts=primary_error.attempts, detail=format_failure(self.primary.name, self.primary.model, primary_error, "primary")) from None


def make_provider(provider: str, gemini_model: str, ollama_model: str) -> LLMProvider:
    if provider == "fake" or (provider == "auto" and not os.getenv("GEMINI_API_KEY")):
        return FakeLLM()
    if provider in {"auto", "gemini"}:
        return ResilientLLM(GeminiLLM(gemini_model), OllamaLLM(ollama_model))
    return OllamaLLM(ollama_model)
