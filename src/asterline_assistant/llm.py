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
OLLAMA_MAX_OUTPUT_TOKENS = 128
OLLAMA_KEEP_ALIVE = "10m"

class ProviderFailure(RuntimeError):
    def __init__(self, message: str, *, error_class: str, transient: bool, status_code: int | None = None, retry_after: float | None = None, quota_type: str | None = None, attempts: int = 0, detail: str | None = None, raw_reply: str | None = None, quality: bool = False) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.transient = transient
        self.status_code = status_code
        self.retry_after = retry_after
        self.quota_type = quota_type
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
    # google-genai may expose RetryInfo only in the rendered exception text.
    # Preserve that server hint without depending on a private SDK response type.
    match = re.search(r"(?:retry[- ]after|retrydelay|retry delay)[^0-9]*(\d+(?:\.\d+)?)\s*s?", str(error), re.IGNORECASE)
    if match:
        return max(0.0, float(match.group(1)))
    return None

def classify_failure(error: Exception, attempts: int = 0) -> ProviderFailure:
    code = _status_code(error)
    raw_error_text = str(error)
    detail = _sanitize(raw_error_text) or "provider request failed"
    config_error = code in CONFIG_CODES or any(marker in detail.lower() for marker in CONFIG_MARKERS)
    retry_after = _retry_after(error)
    is_daily_limit = bool(re.search(r"per[- ]day|daily (?:quota|limit)|day limit|generate_content_free_tier", raw_error_text, re.IGNORECASE))
    is_quota_exhausted = bool(re.search(r"exceeded your current quota|quota exhausted|resource_exhausted", raw_error_text, re.IGNORECASE))
    quota_type = "per-day" if is_daily_limit else ("per-minute" if re.search(r"per[- ]minute|rpm", raw_error_text, re.IGNORECASE) else ("quota" if is_quota_exhausted else None))
    # Exhausted plan/day quotas or long delays (>30s) will not recover during a single UI turn. Avoid
    # burning minutes sleeping on repeated 429s; true short rate-limit 429s still honor the hint.
    exhausted_quota = code == 429 and (
        is_daily_limit
        or (is_quota_exhausted and (retry_after is None or retry_after > 30.0))
        or (retry_after is not None and retry_after > 30.0)
    )
    transient = not config_error and not exhausted_quota and (code in TRANSIENT_CODES or isinstance(error, TimeoutError) or "timeout" in error.__class__.__name__.lower())
    message = "Gemini rejected the model/key: check GEMINI_MODEL and your key." if config_error else "provider request failed"
    return ProviderFailure(message, error_class=error.__class__.__name__, transient=transient, status_code=code, retry_after=retry_after, quota_type=quota_type, attempts=attempts, detail=detail)

def _sanitize(value: str) -> str:
    return re.sub(r"(?i)(api[_ -]?key|token|password|authorization)\s*[:=]?\s*[^\s,;]+", r"\1=<redacted>", value).replace("\n", " ")[:240]

def format_failure(provider: str, model: str, failure: ProviderFailure, role: str) -> str:
    status = f" status={failure.status_code}" if failure.status_code is not None else " status=unknown"
    quota = f" quota={failure.quota_type}" if failure.quota_type else " quota=unknown"
    retry = f" retry_after={failure.retry_after:g}s" if failure.retry_after is not None else " retry_after=none"
    raw = f" raw reply (first 300): {_sanitize(failure.raw_reply)}" if failure.raw_reply is not None else ""
    return f"{role}: {provider}/{model} failed: {failure.error_class}{status}{quota}{retry} {_sanitize(failure.detail)}{raw}"

def _extract_citations(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Normalize common header-style citations, then validate IDs in code."""
    pairs = re.findall(r"\[([^\s—|\]]+)\s+—\s+([^\]]+)\]", text)
    pipe_pattern = re.compile(r"\[\s*(?:SOURCE\s*:?\s*)?([^\s|—\]]+)\s*\|\s*([^\]]+)\]", re.IGNORECASE)
    pipe_pairs = pipe_pattern.findall(text)
    normalized = pipe_pattern.sub(lambda match: f"[{match.group(1)} — {match.group(2).strip()}]", text)
    source_parenthetical = re.compile(r"\(\s*SOURCE\s*:?\s*([^\s|]+)\s*\|\s*([^)]*?)\s*\)", re.IGNORECASE)
    existing = set(pairs + pipe_pairs)
    normalized = source_parenthetical.sub(
        lambda match: "" if (match.group(1), match.group(2).strip()) in existing
        else f"[{match.group(1)} — {match.group(2).strip()}]", normalized,
    )
    bare_source_pattern = re.compile(r"(?im)(?<![\w\[])SOURCE\s*:?\s+([^\s|]+)\s*\|\s*([^\r\n)]+?)\s*$")
    bare_source_pairs = bare_source_pattern.findall(normalized)
    normalized = bare_source_pattern.sub(lambda match: f"[{match.group(1)} — {match.group(2).strip()}]", normalized)
    return normalized, pairs + pipe_pairs + bare_source_pairs

def parse_text_answer(text: str, evidence: EvidencePack, *, attempt: int, degraded: bool) -> AnswerDraft:
    cleaned = re.sub(r"^\s*```(?:json|text)?\s*|\s*```\s*$", "", text or "", flags=re.IGNORECASE | re.DOTALL).strip()
    if not cleaned:
        raise ProviderFailure("model returned an empty reply", error_class="EmptyResponse", transient=False, attempts=attempt, raw_reply=(text or "")[:300])
    cleaned, citations = _extract_citations(cleaned)
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
        self.status_callback: Callable[[str], None] | None = None
        # Optional token-streaming hook: called with each text chunk as it arrives.
        self.stream_callback: Callable[[str], None] | None = None

    def report_status(self, message: str) -> None:
        if self.status_callback is not None:
            self.status_callback(message)

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

    def __init__(self, model: str, *, breaker: CircuitBreaker | None = None, sleep_fn: Callable[[float], None] = time.sleep, random_fn: Callable[[], float] = random.random, max_attempts: int = 5, max_total_wait: float = 45.0) -> None:
        super().__init__()
        self.model = model
        self.breaker = breaker or CircuitBreaker()
        self.sleep_fn = sleep_fn
        self.random_fn = random_fn
        self.max_attempts = max_attempts
        self.max_total_wait = max_total_wait
        self.client: Any = None
        self.stream_callback: Callable[[str], None] | None = None  # token-by-token hook
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
            self.report_status(f"Gemini request · attempt {attempt}/{self.max_attempts}")
            try:
                instruction = prompt if parse_retries == 0 else f"{prompt}\nReturn only a short plain-text answer and valid citations; do not use JSON or Markdown fences."
                # Use streaming if a stream_callback is registered; otherwise batch.
                if self.stream_callback is not None:
                    full_text = ""
                    for chunk in self.client.models.generate_content_stream(model=self.model, contents=instruction, config={"temperature": 0}):
                        token = chunk.text or ""
                        full_text += token
                        if token:
                            self.stream_callback(token)
                    self.breaker.record_success()
                    draft = parse_text_answer(full_text, evidence, attempt=attempt, degraded=False)
                else:
                    response = self.client.models.generate_content(model=self.model, contents=instruction, config={"temperature": 0})
                    self.breaker.record_success()
                    draft = parse_text_answer(response.text or "", evidence, attempt=attempt, degraded=False)
                draft.trace_metadata = {"provider": self.name, "model": self.model, "attempts": attempt, "degraded": False, "error_class": None, "status": None, "quota_type": None, "retry_after": None}
                return draft
            except ProviderFailure as failure:
                if failure.error_class in {"InvalidCitation", "MissingCitation", "EmptyResponse"} and parse_retries == 0:
                    parse_retries += 1
                    continue
                raise failure from None
            except Exception as error:
                failure = classify_failure(error, attempts=attempt)
                if not failure.transient:
                    if failure.status_code in {404, 429} and self.model in {"gemini-2.5-flash", "gemini-3.5-flash"}:
                        self.model = "gemini-3.5-flash-lite"
                        self.report_status("Retrying with gemini-3.5-flash-lite…")
                        continue
                    self.report_status("Gemini quota/configuration limit · switching to local fallback")
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
                self.report_status(f"Gemini busy · retrying in {delay:.0f}s")
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
        self.report_status("Ollama local provider · generating and checking citations")
        context = "\n".join(f"SOURCE {x.chunk.chunk_id} | {x.chunk.title}\n{x.chunk.text}" for x in evidence.chunks)
        example = f"Example citation using an actual source: [{evidence.chunks[0].chunk.chunk_id} — {evidence.chunks[0].chunk.title}]" if evidence.chunks else "If there is no evidence, say you cannot answer."
        errors: list[ProviderFailure] = []
        for attempt in (1, 2):
            self.calls += 1
            try:
                prompt = f"Answer in at most two short sentences using only the evidence. Cite each factual sentence with one exact [id — title] copied from the evidence. If unsupported, say you cannot answer and do not guess. Never combine a refusal with a factual answer. Example: {example}\nQuestion: {question}\nEvidence:\n{context}"
                if attempt == 2:
                    prompt = f"Give at most two short evidence-based sentences. Copy one exact citation from the source headers. If unsupported, abstain. {example} Q: {question}\n{context}"
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
        # Ollama otherwise permits unbounded generation; that makes a weak local
        # model spend far longer than this short, citation-checked answer needs.
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"temperature": 0, "num_predict": OLLAMA_MAX_OUTPUT_TOKENS},
        }).encode()
        request = urllib.request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type": "application/json"})
        # A local 3B model may need several seconds to warm up or answer under CPU load;
        # a short socket timeout turns valid generations into false provider failures.
        with urllib.request.urlopen(request, timeout=45) as response:
            try:
                result = json.loads(response.read())
            except json.JSONDecodeError as error:
                raise ProviderFailure("Ollama response was not valid JSON", error_class="ResponseParseError", transient=False, attempts=1, detail=str(error)) from None
        return str(result.get("response", ""))

    def _parse_text(self, text: str, evidence: EvidencePack, attempt: int) -> AnswerDraft:
        # Direct Ollama use is a normal provider choice; ResilientLLM marks it
        # degraded only when it was used after a primary-provider failure.
        draft = parse_text_answer(text, evidence, attempt=attempt, degraded=False)
        draft.trace_metadata = {"provider": self.name, "model": self.model, "attempts": attempt, "degraded": False, "error_class": None}
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
        self.primary.status_callback = self.status_callback
        self.primary.stream_callback = getattr(self, "stream_callback", None)
        try:
            draft = self.primary.answer(question, evidence)
            self.calls = self.primary.calls
            self.name, self.model = self.primary.name, self.primary.model
            return draft
        except ProviderFailure as primary_error:
            if primary_error.quality:
                return AnswerDraft(answer=primary_error.raw_reply or str(primary_error), error_class=primary_error.error_class, attempts=primary_error.attempts, trace_metadata={"provider": self.primary.name, "model": self.primary.model, "attempts": primary_error.attempts, "degraded": False, "error_class": primary_error.error_class})
            try:
                self.fallback.status_callback = self.status_callback
                draft = self.fallback.answer(question, evidence)
                self.calls = self.primary.calls + self.fallback.calls
                self.name, self.model = self.fallback.name, self.fallback.model
                draft.degraded = True
                draft.attempts = primary_error.attempts + max(1, getattr(draft, "attempts", 1))
                primary_text = format_failure(self.primary.name, self.primary.model, primary_error, "primary")
                draft.primary_error = primary_text
                draft.error_class = primary_error.error_class
                draft.trace_metadata = {"provider": self.name, "model": self.model, "attempts": primary_error.attempts + 1, "degraded": True, "error_class": primary_error.error_class, "primary_error_class": primary_error.error_class, "primary_status": primary_error.status_code, "status": primary_error.status_code, "quota_type": primary_error.quota_type, "retry_after": primary_error.retry_after}
                return draft
            except ProviderFailure as fallback_error:
                self.calls = self.primary.calls + self.fallback.calls
                self.name, self.model = "unavailable", self.fallback.model
                if fallback_error.quality:
                    message = "I couldn't provide a citation-verified answer, so I won't guess. Relevant retrieved sources:"
                else:
                    message = "The language model is temporarily unavailable. Relevant retrieved sources:"
                if evidence.chunks:
                    message += "\n" + "\n".join(f"- [{item.chunk.chunk_id} — {item.chunk.title}]" for item in evidence.chunks)
                else:
                    message += "\n- No relevant corpus sources were retrieved."
                primary_text = format_failure(self.primary.name, self.primary.model, primary_error, "primary")
                fallback_text = format_failure(self.fallback.name, self.fallback.model, fallback_error, "fallback")
                return AnswerDraft(answer=message, degraded=True, provider_error=True, attempts=primary_error.attempts + fallback_error.attempts, error_class=primary_error.error_class, primary_error=primary_text, fallback_error=fallback_text, trace_metadata={"provider": "unavailable", "model": self.fallback.model, "attempts": primary_error.attempts + fallback_error.attempts, "degraded": True, "error_class": primary_error.error_class, "status": primary_error.status_code, "quota_type": primary_error.quota_type, "retry_after": primary_error.retry_after, "fallback_error_class": fallback_error.error_class})

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

def make_provider(provider: str, gemini_model: str, ollama_model: str, *, no_fallback: bool = False) -> LLMProvider:
    if provider == "fake" or (provider == "auto" and not os.getenv("GEMINI_API_KEY")):
        return FakeLLM()
    if provider in {"auto", "gemini"}:
        if no_fallback:
            return GeminiLLM(gemini_model)
        return ResilientLLM(GeminiLLM(gemini_model), OllamaLLM(ollama_model))
    return OllamaLLM(ollama_model)
