# Build progress

- [ ] PRE-STEP — no-fallback eval mode, safe pacing, baseline freeze
  - [ ] inspect prior mixed-provider report
  - [ ] implement flags, retries, resume, and provider counts
  - [ ] run tests and the required baseline no-fallback eval
- [ ] A — improved seven-agent LangGraph pipeline
  - [ ] typed/redacted agent spans
  - [ ] analyst, verifier, and memory-writer integration
  - [ ] tests and commit
- [ ] B — failure-driven improvement and comparative evaluations
  - [ ] cluster baseline failures and write failure log
  - [ ] run improved and memory evaluations
  - [ ] tests and commit
- [ ] C — Streamlit UI and headless coverage
  - [ ] dark dense UI, sessions, chat, evidence, memory panel
  - [ ] AppTest and startup checks
  - [ ] tests and commit
- [ ] D — observability and trace exports
  - [ ] redaction test, optional uploads, representative exports
  - [ ] docs/TRACES.md
  - [ ] tests and commit
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
