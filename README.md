# Asterline Assistant

A reproducible research assistant over the immutable Asterline corpus. `baseline` is the frozen hybrid-retrieval answer path; `improved` adds the explicit seven-agent LangGraph, memory-aware routing, evidence analysis, and verifier boundary.

## Setup

Requires Python 3.11 and uv. In PowerShell or macOS/Linux:

```text
uv sync
uv run asterline ingest
uv run asterline run
```

The assignment accepts a local demonstration; a public deployment is not required.

The committed `uv.lock` pins the core stack used for this submission:

| Package | Locked version |
|---|---:|
| chromadb | 1.5.9 |
| fastembed | 0.9.1 |
| google-genai | 2.33.0 |
| langgraph | 1.2.18 |
| langsmith | 0.14.13 |
| pydantic | 2.14.1 |
| rank-bm25 | 0.2.2 |
| streamlit | 1.66.1 |
| pytest | 9.1.1 |

The first index build downloads `BAAI/bge-small-en-v1.5` once through FastEmbed (approximately 70 MB) into `.cache/fastembed/`; it is cached locally and ignored by Git. No generation model is downloaded by setup. Ollama uses the already-installed `llama3.2:3b` when selected or as Gemini fallback.

Copy `.env.example` to `.env` and set Gemini/LangSmith variables locally. The app loads them with python-dotenv without printing values. Missing credentials select FakeLLM demo mode; LangSmith tracing and uploads skip when unavailable. Gemini and Ollama usage may have provider account or local hardware costs; FakeLLM is offline and free.

On Windows, Ollama normally runs as a local service/application. Confirm it is available with `ollama list`; the expected fallback model is `llama3.2:3b`. Gemini quota or service errors automatically try Ollama. If the local server is unavailable too, the CLI returns a clean source-only response and prints sanitized diagnostics instead of a traceback. `uv run asterline ask "..." --provider ollama` is a direct local-provider smoke test.

## Commands

```text
uv run asterline ask "What is the current Launch ingestion rate?" --provider fake --config baseline --user alice
uv run asterline ask "What is the current Launch ingestion rate?" --provider ollama --config baseline --user alice
uv run asterline eval --provider fake --config baseline --no-fallback
uv run asterline eval-memory --provider fake
uv run asterline export-traces
uv run asterline publish-results
uv run asterline report
uv run pytest
```

PowerShell is the supported Windows entry point. `Makefile` contains equivalent `make setup`, `make run`, `make test`, and `make eval` targets for Mac/Linux.

`--no-fallback` keeps Gemini comparisons pure and retries provider errors on resume. `ASTERLINE_EVAL_MIN_INTERVAL_SECONDS` defaults to 8; `--skip-preflight` is an emergency escape hatch. Answers show provider/model, citations, and degraded status. Provider outages return a clean source-only response, never a traceback.

## Trust boundaries

Every research citation must come from retrieved corpus chunks. Memory is per-user context, never company evidence. Secrets and sensitive personal data are redacted or rejected before storage, transcripts, caches, or traces. The corpus and evaluation datasets are hash-tested and never modified by the app.

See [docs/DESIGN.md](docs/DESIGN.md), [docs/EVAL.md](docs/EVAL.md), [docs/FAILURE_LOG.md](docs/FAILURE_LOG.md), and [docs/DEMO.md](docs/DEMO.md).

The assignment also requires accessible LangSmith runs. See [docs/TRACES.md](docs/TRACES.md) for safe synthetic-run setup, sharing steps, and the reviewer invite alternative. `export-traces` creates local illustrative exports and attempts best-effort dataset/report upload; it does not automatically create public trace URLs. Paste verified real shared URLs into `docs/TRACES.md` before submitting.
