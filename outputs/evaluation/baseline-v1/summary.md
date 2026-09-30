# Executed retrieval baseline — September 30, 2026

80 questions · pinned 41,170-review corpus · CPU · 20 candidates reranked to five evidence slots. No generator calls or human claim-support judgments were made.

| Split | Category | Questions | Game precision@5: bi → rerank | Game coverage@5: bi → rerank | Warm latency p50 / p95 |
|---|---|---:|---|---|---|
| development | comparison | 5 | 68.0% → 72.0% | 60.0% → 80.0% | 2.15s / 2.62s |
| development | game_topic | 20 | 89.0% → 93.0% | 100.0% → 100.0% | 2.30s / 3.72s |
| development | named_game | 10 | 96.0% → 86.0% | 100.0% → 90.0% | 2.10s / 2.98s |
| development | unsupported | 5 | unmeasured → unmeasured | unmeasured → unmeasured | 1.66s / 2.38s |
| holdout | comparison | 5 | 100.0% → 96.0% | 90.0% → 90.0% | 2.26s / 4.40s |
| holdout | game_topic | 20 | 94.0% → 100.0% | 100.0% → 100.0% | 2.14s / 2.60s |
| holdout | named_game | 10 | 98.0% → 100.0% | 100.0% → 100.0% | 2.12s / 2.25s |
| holdout | unsupported | 5 | unmeasured → unmeasured | unmeasured → unmeasured | 2.23s / 3.24s |

## Findings and operating implication

- Both requested games appear in **8 of 10** final comparison evidence sets. The bi-encoder covers both in 6 of 10. This small sample identifies failures; it is not a precise production success-rate estimate.
- In `compare-02`, Terraria/Stardew Valley retrieves five Core Keeper reviews after reranking. Related-game evidence cannot support a requested comparison of the two named games.
- In `compare-06`, the final evidence covers Cities: Skylines II but omits Dragon's Dogma 2. A response should acknowledge missing evidence rather than invent a balanced comparison.
- Development named-game precision falls from 96% to 86% after reranking; reranking is not a universal quality improvement. Holdout named-game precision rises from 98% to 100%.
- A known seed review is recovered in the top five for only **1 of 40 topic questions** at either stage. Anchors are incomplete labels and often have plausible alternative reviews; this does not establish a 2.5% answer-relevance rate. Human pooled relevance judgments are needed.
- The current evidence supports continued use as an inspectable research aid, with particular caution on comparisons and unavailable facts. It does not support a factuality claim or an automated product decision.

## Measurement boundary

Model initialization took 9.08s, excluding artifact downloads. Reported query latency is warmed encode/search plus reranking only; the first warmup, model startup, UI, network and generation are excluded.

Target-game metrics measure requested title coverage. They do not establish topical relevance, citation correctness or claim-level support. Unsupported questions have no proxy accuracy score; their abstention behavior remains unmeasured. The question mix is authored and game-balanced rather than sampled from actual users. No retrieval changes were tuned on this baseline.

[Protocol and human adjudication](../../../evaluation/README.md) · [Per-query records](per_query.json) · [Run manifest](manifest.json)
