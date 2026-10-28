# Failure log

Generated from `eval/results/baseline_fake_20261028T144638Z.json` and `eval/results/improved_fake_20261028T151540Z.json`. Metrics and case rows are read from JSON; no result is hand-entered.

Paired provider/model/config: `fake/deterministic/baseline` → `fake/deterministic/improved`.

| Metric | Before | After |
|---|---:|---:|
| retrieval_recall_at_5 | 0.9744 (78/80 applicable) | 0.9359 (78/80 applicable) |
| mrr | 0.7654 (78/80 applicable) | 0.7049 (78/80 applicable) |
| fact_accuracy | 0.7250 (80/80 applicable) | 0.7750 (80/80 applicable) |
| citation_validity | 1.0000 (80/80 applicable) | 0.9750 (80/80 applicable) |
| citation_precision | 0.5962 (78/80 applicable) | 0.6218 (78/80 applicable) |
| citation_recall | 0.8077 (78/80 applicable) | 0.7714 (78/80 applicable) |
| conflict_handling | 0.4286 (7/80 applicable) | 1.0000 (7/80 applicable) |
| abstention_accuracy | 0.9125 (80/80 applicable) | 0.9125 (80/80 applicable) |
| provider_error_count | 0 | 0 |

## Failed or provider-error cases

| Case | Input | Expected | Before | After | Root cause | Fix / remaining status |
|---|---|---|---|---|---|---|
| — | No failures recorded | — | — | — | — | — |

This log describes measured behavior; a provider-error reduction is not an answer-quality improvement. FakeLLM metrics are not mixed into this comparison.
