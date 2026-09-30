# Retrieval and answer evaluation

**Decision:** how reliably does the live retrieval design supply evidence for the games and questions a visitor requests?

The [versioned benchmark](benchmark.json) contains **80 AI-assisted evaluation questions**, not production user logs: 20 named-game questions, 40 game/topic questions, 10 two-game comparisons and 10 questions whose requested facts are unavailable from this review snapshot. It uses 20 games from the pinned 41,170-review corpus. Each topic question has a traceable source-review ID and text hash. Seed anchors were inspected during authoring, including corrections where a keyword occurred incidentally rather than supporting the requested theme.

The 40 development questions and 40 holdout questions use disjoint game sets. Comparisons stay within their assigned split. The baseline is descriptive, with no retrieval tuning performed. Future tuning should use development questions and a fresh benchmark for final evaluation after holdout results have been inspected.

## Run the measured retrieval baseline

```bash
pip install -r deploy/requirements.txt
python scripts/evaluate_retrieval.py --output outputs/evaluation/new-run
```

The runner downloads/validates the pinned original artifacts, pins both model revisions to the benchmark, exposes the existing bi-encoder top-20 candidates and reranks that **same pool**. It does not change the live ranking policy. It saves progress per question and writes per-query records, aggregate results, warm latency, source/model/configuration hashes and dependency versions.

[Executed baseline summary](../outputs/evaluation/baseline-v1/summary.md) · [Per-query evidence](../outputs/evaluation/baseline-v1/per_query.json) · [Run manifest](../outputs/evaluation/baseline-v1/manifest.json)

## What the metrics mean

| Measure | Denominator / meaning | What it does not prove |
|---|---|---|
| Target-game precision@5 | Of five evidence slots, fraction belonging to requested game(s); absent slots count as misses | Topical relevance or factual correctness |
| Target-game coverage@5 | Requested game titles represented / requested titles | Adequate evidence for every requested theme |
| Source-anchor hit@5 and reciprocal rank | Recovery of one known supporting review; rank searched across the 20-candidate pool | Complete recall: many unlabelled reviews may also answer |
| Warm latency p50/p95 | Timed encode/search plus reranking after model loading and one warmup | Browser, download, network or generation latency |
| Citation numbering | Whether generated bracket IDs refer to supplied evidence positions | Whether the cited review supports the claim |
| Human relevance / claim support | Explicit reviewer judgments with rationale | Currently unmeasured until a reviewer fills the adjudication files |

Source-anchor recovery is intentionally a narrow diagnostic and can be low even when good alternative evidence is retrieved. This benchmark is source-informed and game-balanced rather than representative of live user traffic. Five comparisons per split give limited precision for subgroup estimates. Results do not establish current game quality, model factuality, sentiment accuracy or commercial impact.

## Answer generation and human adjudication

The executed baseline is retrieval-only. **GPT-OSS claim support, citation correctness and abstention have not been measured in this run.** No server-side Groq credential is available to this checkout.

To evaluate generation, configure `GROQ_API_KEY` in your local environment and explicitly run:

```bash
python scripts/evaluate_retrieval.py --generate --output outputs/evaluation/generated-v1
```

The runner calls the same `openai/gpt-oss-120b` model and prompt builder as the app. Generation is never triggered implicitly. It records allow-listed provider diagnostics and leaves every answer's claim-support status `unjudged`.

Each run writes two local adjudication templates, excluded from git to avoid accidentally publishing incomplete reviewer work:

- `retrieval_judgments.csv`: a deduplicated pool from both top-five rankings. Grade each review **0 irrelevant / 1 partially useful / 2 directly useful**, after reading the full review from the pinned corpus. Record reviewer and rationale. A game-name match alone is insufficient for topic questions.
- `answer_judgments.csv`: split an answer into atomic claims; record cited review IDs and grade each **supported / unsupported / contradicted / insufficient context**. Judge abstention separately for unsupported questions. Use only the supplied evidence, including the prompt's 900-character truncation, when assessing support.

Second-review a sample and adjudicate disagreements before presenting an aggregate claim-support score. Do not turn empty judgments into zeros or treat a valid citation number as factual correctness.

## Small ownership exercise

Start with `compare-02`, `compare-06`, and three topic questions. Reconstruct the returned reviews using their `review_id`, grade relevance, and explain whether each failure originates in candidate retrieval, reranking or insufficient coverage. Those judgments are more useful than guessing from similarity scores.
