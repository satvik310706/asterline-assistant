import json

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
