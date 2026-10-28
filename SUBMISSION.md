# Asterline Assistant — Take-home submission

Repository: [PASTE_REPOSITORY_URL_HERE]
LangSmith traces: [PASTE_SHARED_TRACE_LINKS_HERE]

## Executive summary
A research assistant that answers only from an immutable corpus, cites evidence,
handles supersession and abstention, and remembers safe user context across restarts.
The baseline is frozen; the improved configuration adds seven typed agent stages.

## Reproduction
uv sync
uv run pytest -q
uv run asterline ask "What is the current Launch ingestion rate?" --provider fake
uv run asterline run

## Generated evaluation artifacts
baseline_fake.json: provider=fake config=baseline cases=80
  retrieval_recall_at_5: 0.9743589743589743 (78/80 applicable)
  mrr: 0.7653846153846156 (78/80 applicable)
  fact_accuracy: 0.725 (80/80 applicable)
  citation_validity: 1.0 (80/80 applicable)
  citation_precision: 0.5961538461538461 (78/80 applicable)
  citation_recall: 0.8076923076923077 (78/80 applicable)
  conflict_handling: 0.42857142857142855 (7/80 applicable)
  abstention_accuracy: 0.9125 (80/80 applicable)
  fallback_fraction: 0.0 (80/80 applicable)
improved_fake.json: provider=fake config=improved cases=80
  retrieval_recall_at_5: 0.9743589743589743 (78/80 applicable)
  mrr: 0.758974358974359 (78/80 applicable)
  fact_accuracy: 0.7 (80/80 applicable)
  citation_validity: 1.0 (80/80 applicable)
  citation_precision: 0.5833333333333334 (78/80 applicable)
  citation_recall: 0.7756410256410257 (78/80 applicable)
  conflict_handling: 0.42857142857142855 (7/80 applicable)
  abstention_accuracy: 0.9125 (80/80 applicable)
  fallback_fraction: 0.0 (80/80 applicable)
memory_fake.json: provider=fake config=None cases=None

## Evidence and memory guarantees
Citations are parsed and validated against retrieved chunks. Memory is labelled
unverified context and never counts as company evidence. Secrets and sensitive
personal data are redacted or rejected before storage and tracing. SQLite is the
source of truth; hard delete removes memory rows, vectors, sessions, and messages.

## Limitations
FakeLLM reports verify plumbing, not provider quality. Gemini quality runs and
remote LangSmith links require the author's credentials and quota. Deleted data
may remain in already-shipped remote traces. Regex matching is a rough metric.

## AI use disclosure
# AI use disclosure

- Codex (GPT) wrote most code under my direction.
- Claude helped with planning, datasets, and prompts.
- Tests and evaluation runners were used to verify behavior and expose failures.

## Author edits

_Add any additional tools, prompts, or review details here before submission._

## Design excerpt
# Design

```mermaid
flowchart LR
  Q[User question] --> R[router]
  R --> M[memory_reader\nUNVERIFIED USER CONTEXT]
  R --> S[retriever\nhybrid dense + BM25]
  S --> A[evidence_analyst\ncurrent/superseded]
  M --> W[answer_writer]
  A --> W
  W --> V[verifier]
  V --> MW[memory_writer\npolicy gate]
  MW --> DB[(SQLite user memory)]
  A --> C[(Chroma corpus index)]
```

Baseline keeps the original route → retrieve → answer path frozen. Improved uses seven named graph nodes. Corpus records are indexed one-to-one; Chroma and BM25 are fused, while SQLite is the source of truth for user memories, sessions, and messages. Memory is filtered by `user_id` and is never added to citations.

Trade-offs: deterministic policy and validation are easier to explain than a fully autonomous agent, but FakeLLM is intentionally not a quality proxy for Gemini. Regex evaluation is a rough metric. Deleted data may remain in already-shipped LangSmith traces. The current UI is functional rather than polished, and live LangSmith links require credentials.
