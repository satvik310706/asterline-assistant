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
  - Not verified: live `uv run asterline run` health endpoint; the shell could not start the long-running process in this environment
- [x] D — observability and trace exports
  - [x] redaction regression test, optional non-fatal upload hook, representative exports
  - [x] docs/TRACES.md
  - [x] tests passed: tracing redaction 1 passed
  - Not verified: shared LangSmith links or successful dataset upload; the installed client lacked the attempted upload API
- [ ] E — generated documentation and demo
  - [ ] generated EVAL.md, DESIGN.md, README updates, AI_USE.md
  - [ ] DEMO.md and headless demo sequence
  - [ ] tests and commit
- [ ] F — hardening and held-out evaluation
  - [ ] fresh-clone script, secret scan, outage simulations
  - [ ] held-out datasets and honest gap report
  - [ ] tests and commit

## Verification notes

Status is updated after each deliverable with what works, verification commands, and remaining gaps.
