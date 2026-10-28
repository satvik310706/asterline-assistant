# Walkthrough readiness

## One-turn flow

`run_improved_research` builds a seven-node LangGraph: router, memory reader,
retriever, evidence analyst, answer writer, verifier, and memory writer. The
router classifies commands before an LLM call; retrieval combines dense Chroma
and token-preserving BM25; the analyst marks current/superseded evidence; the
writer cites only retrieved corpus chunks; the verifier rejects unsupported
citations and numbers; the memory writer applies the policy gate.

The baseline remains selectable with `ASTERLINE_CONFIG=baseline`. It is the
frozen retrieval/answer path and does not use the improved analyst/verifier.

## Three live-change drills

1. Memory TTL: call `MemoryStore.expire(user, memory_type="task", max_age_days=14)`
   from a maintenance command. The injected `Clock` makes this deterministic;
   add an expiry scenario beside the memory tests.
2. Retrieval parameter: change `retrieve(query, retriever, k=5)` to pass a
   different `k`, or expose it from `Settings`. The retriever contract and
   evidence models remain unchanged, so the change is localized.
3. Conflict rule: add a narrowly scoped marker/scope rule in the analyst
   label function in `src/asterline_assistant/graph.py`, then add a synthetic
   two-chunk test. The verifier still enforces citations independently.

## Explainable trade-offs

The baseline is deliberately preserved for comparison. The improved path uses
deterministic rules before LLM calls to control cost and quota. Memory is
personal context, never evidence. SQLite is the source of truth; vector memory
is disposable. Deleted data can remain in already-shipped remote traces, and
regex-based evaluation is a rough metric rather than a semantic judge.
