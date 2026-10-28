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


class ProviderFailure(RuntimeError):
    def __init__(self, message: str, *, error_class: str, transient: bool, status_code: int | None = None, retry_after: float | None = None, attempts: int = 0) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.transient = transient
        self.status_code = status_code
        self.retry_after = retry_after
        self.attempts = attempts


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
    match = re.search(r"\b(400|401|403|429|500|503|504)\b", str(error))
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
    transient = code in TRANSIENT_CODES or isinstance(error, TimeoutError) or "timeout" in error.__class__.__name__.lower()
    error_class = f"HTTP{code}" if code else error.__class__.__name__
    return ProviderFailure("provider request failed", error_class=error_class, transient=transient, status_code=code, retry_after=_retry_after(error), attempts=attempts)


class LLMProvider:
    name = "unknown"
    model = "unknown"

    def __init__(self) -> None:
        self.calls = 0
        self.last_metadata: dict[str, str | int | bool | None] = {}

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        raise NotImplementedError


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
        prompt = f"Answer only from this evidence. Cite each claim as [chunk_id — title]. If insufficient, abstain.\nQuestion: {question}\nEvidence:\n{context}"
        waited = 0.0
        for attempt in range(1, self.max_attempts + 1):
            self.calls += 1
            try:
                response = self.client.models.generate_content(model=self.model, contents=prompt, config={"temperature": 0, "response_mime_type": "application/json", "response_schema": AnswerDraft})
                self.breaker.record_success()
                draft = AnswerDraft.model_validate(json.loads(response.text or "{}"))
                draft.attempts = attempt
                draft.trace_metadata = {"provider": self.name, "model": self.model, "attempts": attempt, "degraded": False, "error_class": None}
                return draft
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


class OllamaLLM(LLMProvider):
    name = "ollama"

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model

    def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
        self.calls += 1
        context = "\n".join(f"{x.chunk.chunk_id}: {x.chunk.text}" for x in evidence.chunks)
        payload = json.dumps({"model": self.model, "prompt": f"Answer briefly from evidence only. Question: {question}\n{context}", "stream": False, "format": "json"}).encode()
        request = urllib.request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read())
            draft = AnswerDraft.model_validate(json.loads(result["response"]))
            draft.degraded = True
            draft.trace_metadata = {"provider": self.name, "model": self.model, "attempts": 1, "degraded": True, "error_class": None}
            return draft
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
            try:
                draft = self.fallback.answer(question, evidence)
                self.calls = self.primary.calls + self.fallback.calls
                self.name, self.model = self.fallback.name, self.fallback.model
                draft.degraded = True
                draft.attempts = primary_error.attempts + max(1, getattr(draft, "attempts", 1))
                draft.error_class = primary_error.error_class
                draft.trace_metadata = {"provider": self.name, "model": self.model, "attempts": primary_error.attempts + 1, "degraded": True, "error_class": primary_error.error_class}
                return draft
            except ProviderFailure as fallback_error:
                self.calls = self.primary.calls + self.fallback.calls
                self.name, self.model = "unavailable", self.fallback.model
                message = "The language model is temporarily unavailable; here are the most relevant sources:"
                return AnswerDraft(answer=message, degraded=True, provider_error=True, attempts=primary_error.attempts + fallback_error.attempts, error_class=fallback_error.error_class, trace_metadata={"provider": "unavailable", "model": self.fallback.model, "attempts": primary_error.attempts + fallback_error.attempts, "degraded": True, "error_class": fallback_error.error_class})


def make_provider(provider: str, gemini_model: str, ollama_model: str) -> LLMProvider:
    if provider == "fake" or (provider == "auto" and not os.getenv("GEMINI_API_KEY")):
        return FakeLLM()
    if provider in {"auto", "gemini"}:
        return ResilientLLM(GeminiLLM(gemini_model), OllamaLLM(ollama_model))
    return OllamaLLM(ollama_model)
