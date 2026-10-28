from pathlib import Path

from asterline_assistant.config import Settings
from asterline_assistant.corpus import EXPECTED_CORPUS_LINES, EXPECTED_CORPUS_SHA256, corpus_sha256, load_corpus
from asterline_assistant.llm import FakeLLM
from asterline_assistant.memory import MemoryStore
from asterline_assistant.models import EvidencePack
from asterline_assistant.retrieval import TOKEN_RE, tokenize
import json

import pytest

from asterline_assistant.eval_runner import abstention_pass, cache_key, citation_pairs, fact_pass, superseded_pass
from asterline_assistant.llm import AnswerDraft, CircuitBreaker, GeminiLLM, LLMProvider, OllamaLLM, ProviderFailure, ResilientLLM, classify_failure, format_failure, parse_text_answer
from asterline_assistant.models import HistoryMessage
from asterline_assistant.research import route
from asterline_assistant.retrieval import HybridRetriever


ROOT = Path(__file__).parents[1]


def test_corpus_integrity() -> None:
    path = ROOT / "data/corpus.jsonl"
    assert sum(1 for _ in path.open(encoding="utf-8")) == EXPECTED_CORPUS_LINES
    assert corpus_sha256(path) == EXPECTED_CORPUS_SHA256
    assert len(load_corpus(path)) == 75


def test_tokenizer_preserves_operational_tokens() -> None:
    tokens = tokenize("INC-2025-12 2.3.1 429 99.92%")
    assert {"inc-2025-12", "2.3.1", "429", "99.92%"} <= set(tokens)


def test_fake_llm_cites_retrieved_chunks() -> None:
    chunks = load_corpus(ROOT / "data/corpus.jsonl")[:1]
    result = FakeLLM().answer("what is this", EvidencePack(query="what is this", chunks=[{"chunk": chunks[0]}]))
    assert chunks[0].chunk_id in result.cited_chunk_ids
    assert f"[{chunks[0].chunk_id} — {chunks[0].title}]" in result.answer


def test_memory_isolation_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.sqlite3"
    first = MemoryStore(path)
    first.write_explicit("alice", "uses Team")
    first.connection.close()
    second = MemoryStore(path)
    assert [m.content for m in second.list_active("alice")] == ["uses Team"]
    assert second.list_active("bob") == []


def test_memory_filters_secret_and_sensitive_claims(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    assert store.write_explicit("alice", "API key: synthetic-secret") is None
    assert store.write_explicit("alice", "my medical diagnosis is private") is None


def test_memory_forget_correction_and_hard_isolation(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite3")
    assert store.write_explicit("alice", "uses Team") is not None
    assert store.write_explicit("alice", "investigating exports") is not None
    corrected = store.correct("alice", "uses Team", "uses Enterprise")
    assert corrected and corrected.content == "uses Enterprise"
    assert [item.content for item in store.list_active("alice")] == ["investigating exports", "uses Enterprise"]
    assert store.list_active("bob") == []
    assert store.forget("alice", "exports") == 1
    assert all("exports" not in item.content for item in store.list_active("alice"))
    assert store.forget_all("alice") == 2
    assert store.list_active("alice") == []


def test_hybrid_retrieval_uses_one_entry_per_chunk(tmp_path: Path) -> None:
    chunks = load_corpus(ROOT / "data/corpus.jsonl")
    retriever = HybridRetriever(chunks, tmp_path / "chroma")
    results = retriever.search("INC-2025-12 duplicate rows", k=5)
    assert len(results) == 5
    assert any(item.chunk.chunk_id == "incident-12:0" for item in results)


def test_eval_matching_allows_forbidden_fact_only_when_superseded() -> None:
    answer = "The older value was 60 days; the current policy is 20 days and supersedes it."
    assert fact_pass(answer, [r"20 days"], [r"60 days"])
    assert superseded_pass(answer, [r"60 days"])


def test_eval_matching_rejects_unqualified_forbidden_fact() -> None:
    assert not fact_pass("Cold recovery lasts 60 days.", [r"20 days"], [r"60 days"])
    assert not superseded_pass("Cold recovery lasts 60 days.", [r"60 days"])


def test_eval_abstention_requires_decline_and_gold_wording() -> None:
    assert abstention_pass("Enterprise pricing is negotiated and not published.", [r"negotiated"], [])
    assert not abstention_pass("Enterprise costs $999.", [r"not published"], [r"\$999"])


def test_router_rewrites_follow_up_from_history() -> None:
    query = route("And what was it before that?", [HistoryMessage(role="user", content="What is the current AQL timeout?")])
    assert "current AQL timeout" in query
    assert "before that" in query


class FakeAPIError(Exception):
    def __init__(self, code: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"{code} simulated")
        self.code = code
        self.headers = headers or {}


class FakeGeminiClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0

        class Models:
            def __init__(self, owner: "FakeGeminiClient") -> None:
                self.owner = owner

            def generate_content(self, **_: object) -> object:
                self.owner.calls += 1
                outcome = self.owner.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        self.models = Models(self)


def answer_payload(answer: str = "ready") -> object:
    return type("Response", (), {"text": answer})()


def test_gemini_retries_503_with_bounded_backoff() -> None:
    sleeps: list[float] = []
    provider = GeminiLLM("test", sleep_fn=sleeps.append, random_fn=lambda: 0.0)
    provider.client = FakeGeminiClient([FakeAPIError(503), answer_payload()])
    result = provider.answer("q", EvidencePack(query="q", chunks=[]))
    assert result.answer == "ready"
    assert provider.client.calls == 2
    assert sleeps == [1.0]
    assert result.attempts == 2


def test_resilient_provider_reports_live_gemini_retry_status() -> None:
    messages: list[str] = []
    primary = GeminiLLM("test", sleep_fn=lambda _: None, random_fn=lambda: 0.0)
    primary.client = FakeGeminiClient([FakeAPIError(503), answer_payload()])
    fallback = OllamaLLM("local", request_fn=lambda _: "unused")
    provider = ResilientLLM(primary, fallback)
    provider.status_callback = messages.append
    provider.answer("q", EvidencePack(query="q", chunks=[]))
    assert any("Gemini request" in item for item in messages)
    assert any("retrying in" in item for item in messages)


def test_gemini_honors_retry_after() -> None:
    sleeps: list[float] = []
    provider = GeminiLLM("test", sleep_fn=sleeps.append, random_fn=lambda: 0.0)
    provider.client = FakeGeminiClient([FakeAPIError(429, {"Retry-After": "7"}), answer_payload()])
    provider.answer("q", EvidencePack(query="q", chunks=[]))
    assert sleeps == [7.0]


def test_provider_diagnostic_includes_status_quota_and_retry_after() -> None:
    error = FakeAPIError(429, {"Retry-After": "7"})
    error.args = ("quota exceeded per-minute",)
    failure = classify_failure(error, attempts=5)
    rendered = format_failure("gemini", "test", failure, "primary")
    assert "status=429" in rendered
    assert "retry_after=7s" in rendered
    assert "quota=per-minute" in rendered


def test_gemini_does_not_retry_configuration_error() -> None:
    sleeps: list[float] = []
    provider = GeminiLLM("test", sleep_fn=sleeps.append)
    provider.client = FakeGeminiClient([FakeAPIError(400)])
    with pytest.raises(ProviderFailure) as error:
        provider.answer("q", EvidencePack(query="q", chunks=[]))
    assert error.value.status_code == 400
    assert error.value.error_class == "FakeAPIError"
    assert sleeps == []


class StubFallback(LLMProvider):
    name = "ollama"
    model = "llama3.2:3b"

    def __init__(self, failure: bool = False) -> None:
        super().__init__()
        self.failure = failure

    def answer(self, question: str, evidence: EvidencePack):
        self.calls += 1
        if self.failure:
            raise ProviderFailure("offline", error_class="ConnectionError", transient=True, attempts=1)
        return type("Draft", (), {"answer": "fallback answer", "degraded": True, "trace_metadata": {}})()


def failing_gemini() -> GeminiLLM:
    provider = GeminiLLM("test", max_attempts=1)
    provider.client = FakeGeminiClient([FakeAPIError(503)])
    return provider


def test_resilient_provider_falls_back_and_marks_degraded() -> None:
    result = ResilientLLM(failing_gemini(), StubFallback()).answer("q", EvidencePack(query="q", chunks=[]))
    assert result.answer == "fallback answer"
    assert result.degraded is True


def test_double_outage_returns_clean_message_without_traceback() -> None:
    result = ResilientLLM(failing_gemini(), StubFallback(failure=True)).answer("q", EvidencePack(query="q", chunks=[]))
    assert result.provider_error is True
    assert result.answer.startswith("The language model is temporarily unavailable;")
    assert "Traceback" not in result.answer
    assert result.primary_error and "primary: gemini/test failed" in result.primary_error
    assert result.fallback_error and "fallback: ollama/llama3.2:3b failed" in result.fallback_error


def test_gemini_failure_followed_by_ollama_success_keeps_primary_diagnostic() -> None:
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    reply = f"Current answer [{chunk.chunk_id} {chr(0x2014)} {chunk.title}]."
    fallback = OllamaLLM("llama3.2:3b", request_fn=lambda _: reply)
    result = ResilientLLM(failing_gemini(), fallback).answer("q", EvidencePack(query="q", chunks=[{"chunk": chunk}]))
    assert result.degraded is True
    assert result.primary_error and "primary: gemini/test failed" in result.primary_error
    assert result.fallback_error is None


def test_gemini_plain_text_parser_accepts_fenced_reply_shape() -> None:
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    raw_reply = f"```text\nThe current policy is documented here [{chunk.chunk_id} {chr(0x2014)} {chunk.title}].\n```"
    result = parse_text_answer(raw_reply, EvidencePack(query="q", chunks=[{"chunk": chunk}]), attempt=1, degraded=False)
    assert result.answer.startswith("The current policy")
    assert result.cited_chunk_ids == [chunk.chunk_id]


def test_ollama_plain_text_validates_citations_and_retries_parse_failures() -> None:
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    responses = iter(["{malformed", f"Answer [${chunk.chunk_id} — {chunk.title}]"])  # first response must be retried
    provider = OllamaLLM("llama3.2:3b", request_fn=lambda _: next(responses).replace("[$", "["))
    result = provider.answer("q", EvidencePack(query="q", chunks=[{"chunk": chunk}]))
    assert result.cited_chunk_ids == [chunk.chunk_id]
    assert provider.calls == 2


@pytest.mark.parametrize("reply,error_class", [("Answer [missing:0 — Nope].", "InvalidCitation"), ("", "EmptyResponse")])
def test_ollama_rejects_invalid_or_empty_text_after_one_retry(reply: str, error_class: str) -> None:
    provider = OllamaLLM("llama3.2:3b", request_fn=lambda _: reply)
    with pytest.raises(ProviderFailure) as error:
        provider.answer("q", EvidencePack(query="q", chunks=[{"chunk": load_corpus(ROOT / "data/corpus.jsonl")[0]}]))
    assert error.value.error_class == error_class
    assert provider.calls == 2


def test_config_errors_do_not_retry_or_trip_breaker() -> None:
    breaker = CircuitBreaker(threshold=1)
    provider = GeminiLLM("gone", breaker=breaker, sleep_fn=lambda _: pytest.fail("config errors must not sleep"))
    provider.client = FakeGeminiClient([FakeAPIError(404)])
    with pytest.raises(ProviderFailure) as error:
        provider.answer("q", EvidencePack(query="q", chunks=[]))
    assert error.value.status_code == 404
    assert breaker.consecutive_failures == 0


def test_preflight_reports_primary_and_fallback_errors() -> None:
    primary = GeminiLLM("gone", max_attempts=1)
    primary.client = FakeGeminiClient([FakeAPIError(404)])
    fallback = OllamaLLM("llama3.2:3b", request_fn=lambda _: (_ for _ in ()).throw(TimeoutError("offline")))
    with pytest.raises(ProviderFailure) as error:
        ResilientLLM(primary, fallback).preflight(EvidencePack(query="q", chunks=[]))
    assert "primary: gemini/gone failed" in str(error.value)
    assert "fallback: ollama/llama3.2:3b failed" in str(error.value)


def test_preflight_always_429_exhausts_once_without_fallback_retry_loop() -> None:
    sleeps: list[float] = []
    primary = GeminiLLM("busy", max_attempts=4, sleep_fn=sleeps.append, random_fn=lambda: 0.0)
    primary.client = FakeGeminiClient([FakeAPIError(429), FakeAPIError(429), FakeAPIError(429), FakeAPIError(429)])
    fallback = OllamaLLM("llama3.2:3b", request_fn=lambda _: (_ for _ in ()).throw(TimeoutError("offline")))
    with pytest.raises(ProviderFailure) as error:
        ResilientLLM(primary, fallback).preflight(EvidencePack(query="q", chunks=[]))
    assert error.value.status_code == 429
    assert primary.client.calls == 1
    assert len(sleeps) == 0


def test_provider_diagnostics_parse_quota_and_retry_delay_from_genai_text() -> None:
    error = classify_failure(Exception("429 RESOURCE_EXHAUSTED: day limit exceeded; retryDelay: 1108s"))
    assert error.status_code == 429
    assert error.quota_type == "per-day"
    assert error.retry_after == 1108
    rendered = format_failure("gemini", "gemini-test", error, "primary")
    assert "quota=per-day" in rendered
    assert "retry_after=1108s" in rendered


def test_provider_diagnostics_keep_generic_quota_class_when_detail_is_truncated() -> None:
    error = classify_failure(Exception("429 RESOURCE_EXHAUSTED: You exceeded your current quota"))
    assert error.quota_type == "quota"


def test_exhausted_gemini_quota_falls_back_without_long_retry_wait() -> None:
    sleeps: list[float] = []
    provider = GeminiLLM("test", sleep_fn=sleeps.append, max_attempts=5)
    error = FakeAPIError(429)
    error.args = ("429 RESOURCE_EXHAUSTED: You exceeded your current quota",)
    provider.client = FakeGeminiClient([error])
    with pytest.raises(ProviderFailure) as failure:
        provider.answer("q", EvidencePack(query="q", chunks=[]))
    assert failure.value.quota_type == "quota"
    assert provider.client.calls == 1
    assert sleeps == []


def test_abstention_without_citations_is_valid() -> None:
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    provider = GeminiLLM("test", max_attempts=1)
    provider.client = FakeGeminiClient([type("Response", (), {"text": "I am unable to answer this request based on the provided evidence."})()])
    result = provider.answer("q", EvidencePack(query="q", chunks=[{"chunk": chunk}]))
    assert result.abstained is True
    assert result.cited_chunk_ids == []


def test_factual_reply_without_citations_is_missing_citation_after_retry() -> None:
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    provider = GeminiLLM("test", max_attempts=2)
    provider.client = FakeGeminiClient([type("Response", (), {"text": "The answer is 20 days."})(), type("Response", (), {"text": "Still 20 days."})()])
    with pytest.raises(ProviderFailure) as error:
        provider.answer("q", EvidencePack(query="q", chunks=[{"chunk": chunk}]))
    assert error.value.error_class == "MissingCitation"
    assert error.value.quality is True
    assert provider.client.calls == 2


def test_invalid_citation_is_distinct_from_missing_citation() -> None:
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    provider = GeminiLLM("test", max_attempts=2)
    provider.client = FakeGeminiClient([type("Response", (), {"text": "Answer [not-in-pack:0 — Nope]."})(), type("Response", (), {"text": "Answer [not-in-pack:0 — Nope]."})()])
    with pytest.raises(ProviderFailure) as error:
        provider.answer("q", EvidencePack(query="q", chunks=[{"chunk": chunk}]))
    assert error.value.error_class == "InvalidCitation"


def test_ollama_prompt_has_no_literal_placeholder() -> None:
    prompts: list[str] = []
    provider = OllamaLLM("llama3.2:3b", request_fn=lambda prompt: prompts.append(prompt) or "I cannot answer from the evidence.")
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    provider.answer("q", EvidencePack(query="q", chunks=[{"chunk": chunk}]))
    assert prompts and "[chunk_id" not in prompts[0]
    assert "never combine a refusal with a factual answer" in prompts[0]


def test_parser_normalizes_ollama_source_pipe_citation() -> None:
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    evidence = EvidencePack(query="q", chunks=[{"chunk": chunk}])
    draft = parse_text_answer(f"The answer is supported. [SOURCE {chunk.chunk_id} | {chunk.title}]", evidence, attempt=1, degraded=True)
    assert f"[{chunk.chunk_id} — {chunk.title}]" in draft.answer
    assert draft.cited_chunk_ids == [chunk.chunk_id]


def test_parser_normalizes_bare_ollama_source_header_captured_from_real_run() -> None:
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    raw_reply = f"The fact is documented.  SOURCE {chunk.chunk_id} | {chunk.title}"
    draft = parse_text_answer(raw_reply, EvidencePack(query="q", chunks=[{"chunk": chunk}]), attempt=1, degraded=True)
    assert f"[{chunk.chunk_id} — {chunk.title}]" in draft.answer
    assert draft.cited_chunk_ids == [chunk.chunk_id]


def test_parser_deduplicates_ollama_source_suffix() -> None:
    chunk = load_corpus(ROOT / "data/corpus.jsonl")[0]
    raw = f"Grounded answer [{chunk.chunk_id} — {chunk.title}]. (SOURCE {chunk.chunk_id} | {chunk.title})"
    draft = parse_text_answer(raw, EvidencePack(query="q", chunks=[{"chunk": chunk}]), attempt=1, degraded=True)
    assert draft.answer.count(f"[{chunk.chunk_id} — {chunk.title}]") == 1
    assert "SOURCE" not in draft.answer
    assert draft.cited_chunk_ids == [chunk.chunk_id]


def test_circuit_breaker_opens_after_three_failures() -> None:
    breaker = CircuitBreaker(threshold=3, cooldown_seconds=60, clock=lambda: 0.0)
    for _ in range(3):
        provider = GeminiLLM("test", breaker=breaker, max_attempts=1)
        provider.client = FakeGeminiClient([FakeAPIError(503)])
        with pytest.raises(ProviderFailure):
            provider.answer("q", EvidencePack(query="q", chunks=[]))
    provider = GeminiLLM("test", breaker=breaker, max_attempts=1)
    provider.client = FakeGeminiClient([answer_payload()])
    with pytest.raises(ProviderFailure) as error:
        provider.answer("q", EvidencePack(query="q", chunks=[]))
    assert error.value.error_class == "CircuitOpen"
    assert provider.client.calls == 0


def test_citation_lists_use_only_parsed_answer_citations() -> None:
    answer = "Current value [retention:0 — Data retention policy]."
    cited = citation_pairs(answer)
    retrieved = ["retention:0", "security:1"]
    assert cited == [("retention:0", "Data retention policy")]
    assert [chunk for chunk in retrieved if chunk not in {item[0] for item in cited}] == ["security:1"]


def test_eval_cache_key_separates_provider_model_and_config() -> None:
    evidence = EvidencePack(query="q", chunks=[])
    row = {"question": "q"}
    fake_key = cache_key(row, "fake", "deterministic", "baseline", evidence)
    gemini_key = cache_key(row, "gemini", "gemini-test", "baseline", evidence)
    improved_key = cache_key(row, "fake", "deterministic", "improved", evidence)
    assert len({fake_key, gemini_key, improved_key}) == 3


def test_eval_run_survives_provider_error_and_degraded_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from asterline_assistant import eval_runner
    from asterline_assistant.models import CorpusChunk, RetrievedChunk

    class DummyRetriever:
        def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
            chunk = CorpusChunk(chunk_id="test:0", doc_id="test", title="Test source", category="test", owner="test", source_url="test://0", published="20260101", version="1", text="good degraded raise answer ready.")
            return [RetrievedChunk(chunk=chunk)]

    class ScenarioProvider(LLMProvider):
        name = "fake-scenario"
        model = "test"

        def answer(self, question: str, evidence: EvidencePack) -> AnswerDraft:
            if "raise" in question:
                raise ProviderFailure("simulated transient outage", error_class="HTTP503", transient=True, attempts=4)
            if "degraded" in question:
                return AnswerDraft(answer="The language model is temporarily unavailable; here are the most relevant sources:", provider_error=True, degraded=True, error_class="ConnectionError")
            return AnswerDraft(answer="The answer is ready. [test:0 — Test source]", cited_chunk_ids=["test:0"], attempts=1)

    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text("\n".join(json.dumps({"id": case_id, "category": "test", "question": question, "gold_chunks": ["test:0"], "gold_facts": ["ready"], "forbidden_facts": [], "expected": "answer", "notes": ""}) for case_id, question in [("good", "good"), ("degraded", "degraded"), ("raises", "raise")]), encoding="utf-8")
    settings = Settings(corpus_path=ROOT / "data/corpus.jsonl", chroma_path=tmp_path / "chroma", fastembed_cache_path=tmp_path / "fastembed", sqlite_path=tmp_path / "memory.sqlite3", llm_provider="fake")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(eval_runner, "validate_corpus_integrity", lambda path: None)
    monkeypatch.setattr(eval_runner, "HybridRetriever", lambda *args: DummyRetriever())
    monkeypatch.setattr(eval_runner, "make_provider", lambda *args: ScenarioProvider())
    result_path = eval_runner.run(dataset, settings)
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["overall"]["provider_error_count"] == 2
    assert {case["status"] for case in report["case_results"]} == {"ok", "provider_error"}
    assert sum(case["provider_error"] for case in report["case_results"]) == 2

