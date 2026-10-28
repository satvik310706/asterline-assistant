# QA checklist

- [ ] Fresh clone: `uv sync`
- [ ] Dataset hashes and line counts: `uv run pytest tests/test_dataset_integrity.py`
- [ ] Full tests: `uv run pytest -q`
- [ ] Offline smoke: `uv run asterline ask "What is the current Launch ingestion rate?" --provider fake`
- [ ] Baseline eval: `uv run asterline eval --provider fake --config baseline --no-fallback`
- [ ] Improved eval: `uv run asterline eval --provider fake --config improved --no-fallback`
- [ ] Memory eval: `uv run asterline eval-memory --provider fake`
- [ ] UI: `uv run asterline run`, then `http://127.0.0.1:8501/_stcore/health`
- [ ] Optional Gemini: use `--no-fallback`, confirm quota and model diagnostics first.
- [ ] Optional LangSmith: `uv run asterline export-traces`; inspect redaction and agent tree.
