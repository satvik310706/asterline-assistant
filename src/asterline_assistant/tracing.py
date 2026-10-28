"""Optional redacted LangSmith tracing."""

from __future__ import annotations

import os
import json
from functools import wraps
from collections.abc import Callable
from typing import Any, TypeVar

from langsmith import traceable

F = TypeVar("F", bound=Callable[..., Any])
_trace_context: dict[str, str | int | bool | None] = {}


def set_trace_context(metadata: dict[str, str | int | bool | None]) -> None:
    _trace_context.clear()
    _trace_context.update(metadata)


def clear_trace_context() -> None:
    _trace_context.clear()


def _enabled() -> bool:
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return False
    provider = _trace_context.get("provider") or os.getenv("ASTERLINE_LLM_PROVIDER")
    return provider != "fake" and provider != "fake-scenario" or os.getenv("ASTERLINE_TRACE_FAKE") == "1"


def _trace_output(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        output = value.model_dump(mode="json")
        metadata = output.pop("trace_metadata", {})
        return {"output": output, "metadata": metadata}
    return value


def traced(name: str) -> Callable[[F], F]:
    """Trace only when a key is configured; otherwise return a no-op decorator."""
    def decorator(function: F) -> F:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _enabled():
                return function(*args, **kwargs)
            safe_input = {"agent": name, "eval_run_id": None, "case_id": None, "category": None, "config": None, "provider": None, "model": None, "attempts": 0, "degraded": False, **_trace_context}
            if args and isinstance(args[0], str):
                safe_input["text"] = args[0][:2000]
            traced_call = traceable(name=name, run_type="chain", process_outputs=_trace_output)(lambda _: function(*args, **kwargs))
            return traced_call(json.dumps(safe_input, sort_keys=True))

        return wrapper  # type: ignore[return-value]

    return decorator

