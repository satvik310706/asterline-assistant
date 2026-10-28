# Submission readiness audit

This is an evidence-based engineering estimate, not a promise of selection or an official score. The assignment permits a local demo; public deployment is not a requirement.

## Verified

- [x] Immutable data: corpus 75 lines, research 80 cases, memory 34 scenarios; SHA-256 values match the assignment. The corpus and datasets were not edited.
- [x] Both `baseline` and `improved` remain selectable. Current paired offline reports use the same 80-case dataset and FakeLLM provider.
- [x] Memory eval: 34/34 scenarios pass with FakeLLM; all categories 1.0; `isolation_leakage=0`, `secret_leak_count=0`, `sensitive_stored_count=0`, and `forget_success=1.0`.
- [x] Research eval: baseline fact accuracy 0.725; improved 0.775. Improved conflict handling is 7/7 (1.0), versus baseline 3/7 (0.4286). Citation validity is 0.975 improved versus 1.0 baseline; this regression is retained in the report.
- [x] Fixed the real-browser incident turn’s verifier failure with a deterministic incident-ID answer path. Browser check with FakeLLM returned four source citations and completed in 1.20s.
- [x] The UI now honors `ASTERLINE_LLM_PROVIDER=fake`; no-key Gemini selection resolves to FakeLLM rather than silently using Ollama.
- [x] `uv run asterline run` health endpoint returned HTTP 200. The real-browser demo showed FakeLLM mode, the sample-question card, grounded incident answer, and cited evidence.
- [x] Full suite: 100 passed in 377.73s before the final targeted fixes. After those fixes, 42 affected graph/UI/reporting/memory tests passed in 147.19s; the 34-scenario memory runner also passed.
- [x] Generated `eval/final/*.json`, `docs/EVAL.md`, `docs/FAILURE_LOG.md`, `SUBMISSION.md`, and `SUBMISSION.pdf` from evaluation output.
- [x] `uv.lock` is unchanged; `.env` is not tracked; `.gitignore` excludes local caches, databases, and `eval/results/`.

## Rubric estimate

The visible rubric categories supplied in the review image sum to 90 points, although its total is labelled `/100`. These estimates are normalized from those six visible rows; any omitted 10-point category is unscored.

| Rubric area | Estimate |
|---|---:|
| Memory design and dependable behavior | 24/25 |
| Grounded research and answer quality | 14/20 |
| Meaningful multi-agent collaboration | 12/15 |
| Evaluation, failure analysis, improvement | 9/10 |
| Observability | 3/10 |
| Code quality and reproducibility | 7/10 |
| Visible-category total | 69/90 |
| Normalized estimate | 77/100 |

This is not an official grading prediction. FakeLLM metrics establish offline behavior, not Gemini quality.

## Remaining risks / not verified

- A prior no-fallback Gemini run had 62 provider errors out of 80; only 18 cases were answer-quality applicable. No full Gemini evaluation was run after the deterministic incident fix because quota was unavailable.
- LangSmith redaction is unit-tested, but live traces, dataset upload, and reviewer access were not verified; the project’s trace quota had been reached.
- The fresh-clone hardening script passed on an earlier integrated version, but was not rerun after the final code edits. The full suite was not rerun after the last targeted changes; the affected tests and memory dataset were rerun.
- The full documented multi-turn walkthrough was not repeated in the real browser. AppTest covers the 10-turn flow; the live-browser check covered the incident answer in FakeLLM mode.
- The latest Ollama and Gemini reports are provider-labelled historical runs; they are not interchangeable with the current FakeLLM comparison.
