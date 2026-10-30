# Retrieval and answer evaluation

**Decision:** how reliably does the live retrieval design supply evidence for the games and questions a visitor requests?

The [versioned benchmark](benchmark.json) contains **80 authored evaluation questions**, not production user logs: 20 named-game questions, 40 game/topic questions, 10 two-game comparisons and 10 questions whose requested facts are unavailable from this review snapshot. It uses 20 games from the pinned 41,170-review corpus. Each topic question has a traceable source-review ID and text hash. Seed anchors were inspected during authoring, including corrections where a keyword occurred incidentally rather than supporting the requested theme.

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

## Claim-level answer evaluation with an LLM judge

[`scripts/evaluate_answers.py`](../scripts/evaluate_answers.py) judges the generated answers of a run made with `--generate`. It splits each answer into sentence-level claims, shows a second model the claims next to the exact five reviews the generator saw, and records one label per claim: supported, partially supported, unsupported or contradicted. The judge also says whether the answer abstains, meaning it states that the reviews lack what the question asks for.

The [Answer evaluation workflow](../.github/workflows/answer-eval.yml) runs both steps in GitHub Actions. It needs a repository secret named `GROQ_API_KEY` and is started by hand because it spends tokens. Nothing runs locally.

| Reported measure | Meaning |
|---|---|
| Supported rate | Claims labelled supported over all claims, with a bootstrap interval that resamples answers |
| Cited claims correct | Among claims that carry a citation, the share that are supported and cite at least one review the judge named as support |
| Judge abstain flag | The judge's own call on whether an answer declines. It has been wrong when checked by hand, so it is not reported as an abstention rate |
| Answers with an unsupported claim | Count of answers with at least one unsupported or contradicted claim |

Limits that stay attached to every number from this path:

- The labels come from a language model. The default judge is `openai/gpt-oss-20b`, which shares a family with the generator, so errors may be correlated. A different family can be passed with `--judge-model`.
- Each claim is a sentence, not an atomic fact. A sentence with two facts gets one label.
- Judge stability is measured by judging every answer twice with the claims in a different order. High agreement shows consistency, not correctness.
- Correctness needs a person. Every run writes `claim_audit_sample.csv`, 40 claims stratified by judge label with the evidence text. After a reviewer fills `reviewer_label`, `python scripts/evaluate_answers.py --score-audit <file>` reports agreement and kappa with the judge. The file contains review text, so it stays out of git.
- The 80 questions are authored and the unsupported category has 10 questions, so its intervals are wide.

## Answer evaluation results, 2026-10-30

Generator `openai/gpt-oss-120b` with the production prompt. Judge `openai/gpt-oss-20b`. All 80 benchmark questions have an answer. 77 have a judged verdict. The other 3 (two game-topic, one named-game) got judge replies that were not valid JSON after two attempts, so they are missing for a format reason and not a rate limit. [Summary](../outputs/evaluation/generated-v1/answer_eval_summary.md) · [Per-claim records](../outputs/evaluation/generated-v1/answer_eval.json).

| Category | Answers judged | Claims | Supported (95% CI) | Cited claims |
|---|---:|---:|---|---:|
| game_topic | 38 of 40 | 198 | 87.9% (82 to 93) | 61% |
| named_game | 19 of 20 | 116 | 88.8% (83 to 94) | 64% |
| comparison | 10 of 10 | 51 | 86.3% (73 to 96) | 59% |
| unsupported | 10 of 10 | 16 | see below | 0% |
| all | 77 of 80 | 381 | 88.5% (85 to 92) | 59% |

- **Claim support.** 88.5% supported, 10.8% partially supported, 0% unsupported, 0.8% contradicted. The 3 contradicted claims were read against the evidence by hand. Two are a correct statement that every retrieved review is about Core Keeper, and one is a statement that most reviews support. None is a contradiction.
- **Citations.** 59% of claims carry a citation. The other 41% are summary sentences with no marker, which is the generator's behaviour and the largest weakness in this evaluation. Among cited claims, 100% were supported and overlapped the reviews the judge named as support. That rate says the cited reviews are the right ones. It does not say that every sentence is cited.
- **Unanswerable questions.** All 10 answers decline without inventing the requested fact (a price, a sales forecast, an FPS figure, email addresses, a release date, an uptime percentage, a budget, a refund, a medical claim, a fictional game). This is a hand reading of the 10 answers by Claude, not the judge. The judge's abstain flag was true for 1 of 10 and is wrong on the other 9, so the harness no longer calls it an abstention rate. Ten questions is a small sample, so a 10 of 10 rate has a wide interval.
- **Judge stability.** 12 answers were judged twice with the claims in a different order. Label agreement was 89.2% with kappa 0.31. Agreement is high because almost every label is supported, and kappa shows how little the judge disagrees with itself beyond that.
- **What this does not show.** Whether a supported claim is a fair summary of the whole corpus. Whether the judge, which shares a family with the generator, favours the generator's phrasing. Any human judgment. The model audit below found the judge stricter than a second reader, which suggests 88.5% is conservative, but it was one reviewer on 40 claims.

## Audit of the judge by a second model, 2026-10-29

A language model other than the judge read 40 claims against the five reviews each answer saw and labelled them with the same four labels. The sample was stratified by judge label, so it over-represents rare labels and its agreement figures do not describe all claims. Labels, reasons and the agreement file are in [`audit/`](audit/claim_audit_v1.csv). The reviewer was Claude, not a person, so this is a cross-check and not the human audit.

| Judge label | Claims audited | Reviewer agreed |
|---|---:|---:|
| supported | 18 | 17 |
| partially supported | 18 | 10 |
| contradicted | 3 | 0 |
| unsupported | 1 | not a claim |

- Agreement on the 39 claims both labelled was 69.2% with kappa 0.43.
- 11 of the 12 disagreements are the judge being stricter than the reviewer. Eight partially supported labels the reviewer called supported, and all three contradicted labels were wrong: two were correct abstentions and one was a claim that most reviews supported.
- The one claim the judge called unsupported was a title fragment, not a claim.
- The reviewer was lenient on summaries that combine facts from several reviews when each fact appears somewhere in the five. A stricter reading would move some supported labels to partially supported.
- One reviewer, one pass, no second opinion, and 18 supported claims out of 283. Treat the judge as conservative and noisy.

The audit also exposed two bugs in this harness that are now fixed. Citations written with fullwidth brackets were read as uncited, which affected 53 of 324 claims. A title fragment was counted as a claim.

## Small ownership exercise

Start with `compare-02`, `compare-06`, and three topic questions. Reconstruct the returned reviews using their `review_id`, grade relevance, and explain whether each failure originates in candidate retrieval, reranking or insufficient coverage. Those judgments are more useful than guessing from similarity scores.
