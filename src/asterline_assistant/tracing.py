"""Optional redacted LangSmith tracing."""

from __future__ import annotations

import os
import json
import hashlib
import re
from functools import wraps
from collections.abc import Callable
from typing import Any, TypeVar

from langsmith import traceable

from .memory import redact_sensitive

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
        # Error payloads can contain SDK bodies or model replies. Keep only
        # class/status identifiers in traces; full details stay out of LangSmith.
        primary_error = output.pop("primary_error", None)
        fallback_error = output.pop("fallback_error", None)
        metadata.pop("primary_error", None)
        if primary_error or output.get("error_class"):
            metadata.setdefault("primary_error_class", output.get("error_class"))
            metadata.setdefault("primary_status", output.get("error_status"))
        metadata.pop("fallback_error", None)
        if fallback_error and "fallback_error_class" not in metadata:
            match = re.search(r"failed:\s*([A-Za-z][A-Za-z0-9_]*)", fallback_error)
            if match:
                metadata["fallback_error_class"] = match.group(1)
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
                safe_input["text"] = redact_sensitive(args[0])[:2000]
            traced_call = traceable(name=name, run_type="chain", process_outputs=_trace_output)(lambda _: function(*args, **kwargs))
            return traced_call(json.dumps(safe_input, sort_keys=True))

        return wrapper  # type: ignore[return-value]

    return decorator

