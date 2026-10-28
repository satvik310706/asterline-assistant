import json

from asterline_assistant import tracing


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
