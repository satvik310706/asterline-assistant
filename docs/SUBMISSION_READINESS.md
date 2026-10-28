# Submission readiness audit

This is an evidence-based engineering estimate, not a promise of selection or an official score. The assignment permits a local demo; public deployment is not a requirement.

## Verified

- [x] Immutable data: corpus 75 lines, research 80 cases, memory 34 scenarios; SHA-256 values match the assignment. The corpus and datasets were not edited.
- [x] Both `baseline` and `improved` remain selectable. The latest improved FakeLLM run covers all 80 research cases; a fresh baseline run was not performed after the FakeLLM and matching logic changes, so no new baseline delta is claimed.
- [x] Memory eval: 34/34 scenarios pass with FakeLLM; all categories 1.0; `isolation_leakage=0`, `secret_leak_count=0`, `sensitive_stored_count=0`, and `forget_success=1.0`.
- [x] Latest improved FakeLLM eval: fact accuracy 1.000 (80/80), citation validity 1.000 (80/80), citation precision 0.851 (77 applicable), citation recall 0.944 (77 applicable), MRR 0.771 (78 applicable), conflict handling 1.000 (7 applicable), and abstention accuracy 1.000 (80/80). These are offline deterministic checks, not Gemini quality results.
- [x] Fixed the real-browser incident turn’s verifier failure with a deterministic incident-ID answer path. Browser check with FakeLLM returned four source citations and completed in 1.20s.
- [x] The UI now honors `ASTERLINE_LLM_PROVIDER=fake`; no-key Gemini selection resolves to FakeLLM rather than silently using Ollama.
- [x] `uv run asterline run` health endpoint returned HTTP 200. The real-browser demo showed FakeLLM mode, the sample-question card, grounded incident answer, and cited evidence.
- [x] Full suite: 100 passed in 377.73s on an earlier revision. After the latest targeted changes, 27 focused provider/graph/eval/reporting tests passed. The full suite was not rerun after the final edits.
- [x] Generated `eval/final/*.json`, `docs/EVAL.md`, `docs/FAILURE_LOG.md`, `SUBMISSION.md`, and `SUBMISSION.pdf` from evaluation output.
- [x] `uv.lock` is unchanged; `.env` is not tracked; `.gitignore` excludes local caches, databases, and `eval/results/`.

## Rating and scope

The visible rubric categories supplied in the review image sum to 90 points, although its total is labelled `/100`. The previous 77/100 figure was an engineering estimate based on an earlier revision, not an official score, and has not been recalculated. The improved offline eval now passes deterministic fact and abstention checks, but that alone cannot justify a 100/100 prediction.

| Latest improved FakeLLM check | Result |
|---|---:|
| Fact accuracy | 80/80 |
| Citation validity | 80/80 |
| Conflict handling | 7/7 applicable |
| Abstention accuracy | 80/80 |
| Citation precision | 0.851 (77 applicable) |
| Citation recall | 0.944 (77 applicable) |
| MRR | 0.771 (78 applicable) |

This is not an official grading prediction. FakeLLM metrics establish offline behavior, not Gemini quality.

## Remaining risks / not verified

- A prior no-fallback Gemini run had 62 provider errors out of 80; only 18 cases were answer-quality applicable. No full Gemini evaluation was run after the deterministic incident fix because quota was unavailable.
- LangSmith redaction is unit-tested, but live traces, dataset upload, and reviewer access were not verified; the project’s trace quota had been reached.
- The fresh-clone hardening script passed on an earlier integrated version, but was not rerun after the final code edits. The full suite was not rerun after the last targeted changes; the affected tests and memory dataset were rerun.
- The full documented multi-turn walkthrough was not repeated in the real browser. AppTest covers the 10-turn flow; the live-browser check covered the incident answer in FakeLLM mode.
- The latest Ollama and Gemini reports are provider-labelled historical runs; they are not interchangeable with the current improved FakeLLM report. The older baseline FakeLLM report is not a fresh paired comparison.
