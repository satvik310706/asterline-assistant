# Asterline Assistant

Phase 1 provides the frozen `baseline` research path: hybrid local retrieval, a cited answer, deterministic abstention when retrieved evidence has no query overlap, and SQLite-backed user memory. The improved analyst/verifier path and Streamlit UI are explicit later-phase work.

## Setup

Windows PowerShell:

```powershell
uv sync
uv run asterline ingest
```

The first embedding initialization downloads and caches `BAAI/bge-small-en-v1.5` through FastEmbed. The download is approximately 130–150 MB and is stored in the user cache, not this repository. Chroma indexes are stored under `data/chroma/` and ignored by Git.

The exact resolved versions are recorded in `uv.lock`. Phase 1 was verified with Python 3.11.9 and these direct packages: Chroma 1.5.9, FastEmbed 0.9.1, rank-bm25 0.2.2, Streamlit 1.66.1, LangGraph 1.2.18, Pydantic 2.14.1, pytest 9.1.1, google-genai 2.33.0, and LangSmith 0.14.13.

## Commands

```powershell
uv run asterline ingest
uv run asterline run
uv run asterline test
uv run asterline eval
```

`eval` expects `eval/datasets/research.jsonl` and writes timestamped metrics to `eval/results/`. It uses FakeLLM and a response cache, so evaluation works without keys or network. `ASTERLINE_CONFIG=baseline` is the default; `improved` is reserved for a later phase.

Gemini is selected explicitly with `ASTERLINE_LLM_PROVIDER=gemini` and the existing `GEMINI_API_KEY`/`GEMINI_MODEL` environment variables. Without a key, `auto` uses FakeLLM. LangSmith tracing is a silent no-op when no LangSmith key is configured. No credentials are printed or stored by the application.

## Phase 1 limitations

The command entry point is functional, but the Streamlit chat UI, full multi-agent graph, conflict/supersession analyst, verifier/repair loop, provider retries/fallback, and complete evaluation datasets are deferred. The supplied corpus is immutable and protected by its expected 75-line SHA-256 test.
