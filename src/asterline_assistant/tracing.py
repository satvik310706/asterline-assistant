"""Optional redacted LangSmith tracing."""

from __future__ import annotations

import os
import json
from functools import wraps
from collections.abc import Callable
from typing import Any, TypeVar

from langsmith import traceable

F = TypeVar("F", bound=Callable[..., Any])


def traced(name: str) -> Callable[[F], F]:
    """Trace only when a key is configured; otherwise return a no-op decorator."""
    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        return lambda function: function

    def decorator(function: F) -> F:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            safe_input = {"agent": name}
            if args and isinstance(args[0], str):
                safe_input["text"] = args[0][:2000]
            traced_call = traceable(name=name, run_type="chain")(lambda _: function(*args, **kwargs))
            return traced_call(json.dumps(safe_input, sort_keys=True))

        return wrapper  # type: ignore[return-value]

    return decorator

