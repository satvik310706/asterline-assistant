# Asterline Assistant

Phase 1 provides the frozen `baseline` research path: hybrid local retrieval, a cited answer, deterministic abstention when retrieved evidence has no query overlap, and SQLite-backed user memory. The improved analyst/verifier path and Streamlit UI are explicit later-phase work.

## Setup

Windows PowerShell:

```powershell
uv sync
uv run asterline ingest
```

The first embedding initialization downloads and caches `BAAI/bge-small-en-v1.5` through FastEmbed. The one-time download is approximately 70 MB and is stored in the project-local `.cache/fastembed/` folder, which is ignored by Git. Chroma indexes are stored under `data/chroma/` and ignored by Git.

The exact resolved versions are recorded in `uv.lock`. Phase 1 was verified with Python 3.11.9 and these direct packages: Chroma 1.5.9, FastEmbed 0.9.1, rank-bm25 0.2.2, Streamlit 1.66.1, LangGraph 1.2.18, Pydantic 2.14.1, pytest 9.1.1, google-genai 2.33.0, and LangSmith 0.14.13.

## Commands

```powershell
uv run asterline ingest
uv run asterline run
uv run asterline test
uv run asterline eval
uv run asterline ask "What is the current Launch ingestion rate?" --provider fake --user alice
```

`eval` expects `eval/datasets/research.jsonl` and writes timestamped metrics to `eval/results/`. It uses FakeLLM and a response cache, so evaluation works without keys or network. `ASTERLINE_CONFIG=baseline` is the default; `improved` is reserved for a later phase.

Configuration calls `python-dotenv`'s `load_dotenv()` so the app loads `GEMINI_API_KEY`, `GEMINI_MODEL`, `OLLAMA_MODEL`, and `LANGSMITH_*` from `.env` without printing or logging their values. Gemini is selected explicitly with `--provider gemini` or `ASTERLINE_LLM_PROVIDER=gemini`. Without a key, `auto` uses FakeLLM. LangSmith tracing is a silent no-op when no LangSmith key is configured. No credentials are printed or stored by the application.

`ask` prints the answer, provider/model, and retrieved citation IDs/titles. The evaluator reports overall and per-category recall@5, MRR, fact accuracy, citation validity, conflict handling, abstention accuracy, latency, and LLM calls, followed by every failed case.

Gemini retries transient 429/500/503/504 and timeout failures with bounded exponential backoff and honors `Retry-After`. Three consecutive exhausted transient failures open a 60-second circuit breaker. Failed Gemini calls fall back to Ollama and are marked degraded; if both providers fail, the CLI returns a clean source-only message rather than a traceback. Set `ASTERLINE_EVAL_DELAY_SECONDS` to throttle uncached evaluation calls.

## Phase 1 limitations

The command entry point is functional, but the Streamlit chat UI, full multi-agent graph, conflict/supersession analyst, verifier/repair loop, and complete evaluation datasets are deferred. The supplied corpus is immutable and protected by its expected 75-line SHA-256 test.
