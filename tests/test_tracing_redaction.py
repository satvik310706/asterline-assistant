import json

import pytest

from asterline_assistant import tracing
from asterline_assistant.models import ResearchResponse


def test_trace_input_redacts_secret(monkeypatch) -> None:
    captured = []

    class FakeTrace:
        def __call__(self, value):
            captured.append(value)
            return "ok"

    monkeypatch.setattr(tracing, "_enabled", lambda: True)
    monkeypatch.setattr(tracing, "traceable", lambda **kwargs: lambda fn: FakeTrace())
    tracing.clear_trace_context()

    @tracing.traced("router")
    def route_for_test(message: str):
        return message

    route_for_test("remember password=super-secret and sk-test123")
    assert captured
    payload = json.loads(captured[0])
    assert "super-secret" not in json.dumps(payload)
    assert "sk-test123" not in json.dumps(payload)
    assert "[REDACTED]" in json.dumps(payload)


def test_trace_output_keeps_error_classes_but_drops_raw_replies() -> None:
    response = ResearchResponse(
        answer="source-only", evidence=[], provider="unavailable", model="local",
        error_class="ClientError", error_status=429,
        primary_error="primary error with raw reply secret-one",
        fallback_error="fallback failed: MissingCitation raw reply secret-two",
        trace_metadata={"primary_error": "raw reply secret-one", "fallback_error": "raw reply secret-two", "error_class": "ClientError", "status": 429},
    )
    payload = tracing._trace_output(response)
    encoded = json.dumps(payload)
    assert "secret-one" not in encoded and "secret-two" not in encoded
    assert "raw reply" not in encoded
    assert payload["metadata"]["primary_error_class"] == "ClientError"
    assert payload["metadata"]["fallback_error_class"] == "MissingCitation"


def test_tracing_does_not_execute_a_failing_node_twice(monkeypatch) -> None:
    calls = 0

    def transparent_traceable(**kwargs):
        return lambda function: lambda value: function(value)

    monkeypatch.setattr(tracing, "_enabled", lambda: True)
    monkeypatch.setattr(tracing, "traceable", transparent_traceable)

    @tracing.traced("memory_writer")
    def write_once(_: str) -> None:
        nonlocal calls
        calls += 1
        raise ValueError("simulated node failure")

    with pytest.raises(ValueError, match="simulated node failure"):
        write_once("synthetic")
    assert calls == 1


def test_trace_upload_failure_after_success_preserves_result_without_retry(monkeypatch) -> None:
    calls = 0

    def failing_trace_upload(**kwargs):
        def decorate(function):
            def traced_call(value):
                result = function(value)
                raise RuntimeError("simulated trace upload failure")
            return traced_call
        return decorate

    monkeypatch.setattr(tracing, "_enabled", lambda: True)
    monkeypatch.setattr(tracing, "traceable", failing_trace_upload)

    @tracing.traced("memory_writer")
    def write_once(_: str) -> str:
        nonlocal calls
        calls += 1
        return "saved"

    assert write_once("synthetic") == "saved"
    assert calls == 1


def test_explicit_tracing_disable_wins_even_if_credentials_exist(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "synthetic-test-value")
    monkeypatch.setenv("ASTERLINE_DISABLE_TRACING", "1")
    assert not tracing._enabled()
