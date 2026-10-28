# Build progress

- [x] PRE-STEP — no-fallback eval mode, safe pacing, baseline freeze
  - [x] inspected prior mixed-provider report: S09/S10 were Gemini ClientError 429 fallbacks
  - [x] implemented flags, retries, resume identity, pacing, and provider counts
  - [x] focused tests passed; required real no-fallback commands were blocked by Gemini preflight
  - Not verified: a full 80-case no-fallback Gemini report because preflight failed before cases started
- [x] A — improved seven-agent LangGraph pipeline
  - [x] typed/redacted agent spans
  - [x] analyst, verifier, and memory-writer integration
  - [x] graph tests and improved FakeLLM smoke run passed
  - Not verified: live LangSmith rendering; credentials were not inspected or required
- [x] B — failure-driven improvement and comparative evaluations
  - [x] clustered the available generated baseline failures and rendered [FAILURE_LOG.md](./FAILURE_LOG.md)
  - [x] ran full 80-case FakeLLM baseline and improved evaluations; improved memory-query pollution was reduced
  - [x] focused graph, corpus, and memory tests passed
  - Not verified: full Gemini baseline/improved comparison or Gemini memory run; Gemini preflight was unavailable
- [x] C — Streamlit UI and headless coverage
  - [x] dark dense UI, sessions, chat, evidence, memory panel
  - [x] AppTest startup, memory actions, FakeLLM remember, and user isolation passed
  - [x] UI test suite passed: 3 passed
  - [x] Live `uv run asterline run` health endpoint returned HTTP 200; real-browser FakeLLM incident answer and citations verified.
- [x] D — observability and trace exports
  - [x] redaction regression test, optional non-fatal upload hook, representative exports
  - [x] docs/TRACES.md
  - [x] tests passed: tracing redaction 1 passed
  - Not verified: shared LangSmith links or successful dataset upload; the installed client lacked the attempted upload API
- [x] E — generated documentation and demo
  - [x] generated EVAL.md, DESIGN.md, README updates, AI_USE.md
  - [x] DEMO.md and documented headless demo sequence
  - [x] published curated FakeLLM reports and reporting test added
  - Not verified: Gemini headline reports, because Gemini preflight was unavailable
- [x] F — hardening and held-out evaluation
  - [x] fresh-clone script, dataset integrity tests, and secret-scan harness
  - [x] 12 held-out research cases and 6 held-out memory scenarios run with FakeLLM
  - [x] held-out results: research fact 0.75, abstention 0.8333; memory all 1.0, isolation 0, secret leaks 0
  - Not verified: fresh-clone harness execution (it stopped on dataset byte-hash mismatch in the Windows test sandbox) and a live Gemini held-out run

## Verification notes

Status is updated after each deliverable with what works, verification commands, and remaining gaps.

## Phase 5 handoff progress

- [x] 5.1 dataset byte preservation: `.gitattributes` added; active workspace raw hashes and line counts match the assignment.
- [x] 5.2 provider diagnostics: status, quota class, Retry-After, model, and sanitized primary/fallback errors are carried through CLI/eval/trace metadata; Google retry-delay text is decoded without exposing secrets.
- [ ] 5.3 full real Gemini baseline/improved/memory runs and a measured quality improvement.
  - [x] Real Gemini quota failure exercised end-to-end; verified fallback to local Ollama succeeds with a grounded answer and validated citation.
  - [ ] Full evaluations remain unverified because Gemini returns HTTP 429 `RESOURCE_EXHAUSTED`; do not interpret FakeLLM metrics as provider quality.
- [x] 5.4 LangSmith upload compatibility, redacted live export, synthetic export labels, and demo-trace runner implemented.
  - Not verified: successful remote upload/live trace capture; external credentials and Gemini quota were unavailable for a clean end-to-end run.
- [x] 5.5 live Streamlit health check returned HTTP 200; fresh browser showed FakeLLM mode and a verified cited incident response.
- [x] 5.6 full repository pytest verification: 45 passed in 309.96s.
  - Fresh-clone hardening later passed: `47 passed`, dataset hashes, FakeLLM smoke, and secret scan (synthetic fixture paths excluded explicitly).
  - Not verified: live Gemini/LangSmith remote runs remain environment/account dependent.

## Phase 6 walkthrough readiness

- [x] Added `docs/WALKTHROUGH.md` with agent flow, trade-offs, and three live-change drills.
- [x] Added `docs/QA_CHECKLIST.md` with fresh-clone, offline, eval, UI, and optional-provider checks.
- [x] Implemented and tested injectable-clock task/project expiry; retrieval `k` remains an explicit typed drill point.

## Phase 7 submission package

- [x] Added `uv run asterline build-submission`, which reads `eval/final/*.json` and generates `SUBMISSION.md` and `SUBMISSION.pdf`.
- [x] Added a generation test and the confirmed repository URL; shared trace links remain explicit because live LangSmith access was unavailable.
- [x] Fresh-clone hardening rerun passed after the synthetic-fixture scanner correction.

## Reliability and UI follow-up

- [x] Verified local Ollama model `llama3.2:3b` is present and direct CLI answers work without downloading a generation model.
- [x] Verified Gemini HTTP 429 → Ollama fallback returns a cited answer; improved parser normalizes bare `SOURCE id | title` headers and validates IDs against evidence.
- [x] UI now streams actual LangGraph node-completion events and elapsed timings; answer text appears only after citation/grounding verification (model-token streaming is intentionally disabled because it would expose unverified text).
- [x] Memory content is HTML-escaped before rendering in the memory panel and memory-used section.
- [x] Ten-turn end-to-end conversation and UI regression tests passed after the UI safety changes.
- [x] Full repository test suite after the latest UI safety changes: `65 passed in 104.61s`.
- [x] Final fresh-clone hardening on the integrated implementation: 52 passed, raw dataset hashes matched, FakeLLM smoke passed, and the non-fixture secret scan passed.

## Submission readiness audit continuation

- [x] Added [SUBMISSION_READINESS.md](./SUBMISSION_READINESS.md) with evidence-based checklist, blockers, and explicit local-demo scope.
- [x] Reporting/submission generation understands memory scenario JSON; tests cover 34 scenarios, category denominators, and precision/recall rendering.
- [x] Partial reports are excluded from `publish-results`; tests ensure an incomplete latest run cannot replace a complete report.
- [x] Research eval cases now use fresh temporary SQLite memory stores; case memories cannot leak across cases or touch the app DB. Cache revisions were bumped to invalidate contaminated answers.
- [x] Tracing wrapper no longer re-executes nodes after a trace upload exception; trace-off env flags support quota-safe local QA.
- [x] Direct Ollama use is no longer mislabeled as a fallback; local generation timeout raised to 45 seconds and covered by a test.
- [x] Added deterministic evidence-backed calculator for seat/annual/overage/retention/SLA arithmetic; calculator refuses unsupported Enterprise overage values and missing evidence.
- [x] Improved evidence analyst honors explicit supersession, requested historical version, and plan/region scope; verifier checks citation ID/title and numeric grounding, with one repair loop.
- [x] UI now marks only context actually used as “Memory used this turn”; memory HTML is escaped and reduced-motion preference is honored.
- [x] Focused verification after these changes: 66 passed, then 14 passed in the added eval-isolation/report/UI regression set.
- [x] Full isolated-memory baseline Ollama eval completed: 80 cases, 0 provider errors; `eval/results/baseline_ollama_20261028T130429Z.json`.
- [x] Full paired 80-case FakeLLM reports regenerated after cache revision; baseline fact accuracy 0.725, improved 0.775, improved conflict handling 7/7.
- [x] Full 34-scenario memory FakeLLM report regenerated after the restart-follow-up fix: 34/34 passed; isolation, secret, and sensitive-data leak counts are zero.
- [x] Limited memory reports are excluded from submission publishing unless scenario count matches `eval/datasets/memory.jsonl`.
- [x] Fixed the incident-ID verifier failure path with deterministic evidence text and citations; browser FakeLLM sample completed in 1.20s.
- [x] UI provider default now honors `ASTERLINE_LLM_PROVIDER=fake` and selects FakeLLM when Gemini credentials are absent.
- [x] Regenerated `docs/EVAL.md`, `docs/FAILURE_LOG.md`, `SUBMISSION.md`, and `SUBMISSION.pdf` from curated evaluation results.
- [x] Full suite previously passed: 100 tests in 377.73s. After the final patches, 42 affected tests passed in 147.19s; memory dataset: 34/34.
- [x] Live local health check returned HTTP 200; browser showed the FakeLLM demo and a cited incident answer.
- [ ] Fresh-clone hardening was not rerun after the final patches; no full Gemini post-fix eval or live LangSmith upload/link was available.
