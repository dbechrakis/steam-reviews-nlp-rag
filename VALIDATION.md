# Validation record

Latest structural review: 2026-09-22 (UTC).

Checked saved CSV arithmetic, sample sizes and evaluation implementation; no retraining or live generation.

Historical records are not labelled as freshly reproduced results.

## September 2026 product-structure verification

- Reorganized the application into an installable `src/steam_review_rag` package.
- Preserved the Community Cloud entry point and root `app.py` compatibility path.
- Added focused tests for artifact integrity, evidence-prompt construction, and safe provider diagnostics.
- Revalidated all committed Python syntax, notebook structure, and saved result arithmetic.
- Installed the exact pinned CPU deployment dependencies under Python 3.12.
- Downloaded and hash-verified the three pinned retrieval artifacts again.
- Loaded the real 41,170-row corpus and matching FAISS index through Streamlit AppTest; the refactored deployment entry point started without exceptions and exposed the chat input.
- Did not download ML models, call Groq, retrain models, or modify recorded model results during this structural review.

## Streamlit deployment preparation

- Installed the pinned CPU app dependencies under Python 3.12.
- Downloaded and hash-verified all three original retrieval artifacts.
- Loaded 41,170 corpus rows, 241 game names and 41,170 FAISS vectors.
- Streamlit AppTest: entrypoint starts with no errors and exposes the chat input.
- AppTest with synthetic retrieval and a simulated provider failure: evidence
  remains available; no app exception. This is a failure-path test, not a model evaluation.
- Integrity checks reject incorrect sizes and hashes; evidence CI passes locally.
- Full semantic search remains unverified in this environment because model
  downloads encountered network timeouts. Groq generation is not tested without
  the owner's key. Community Cloud resource limits must be checked after deploy.

## Public deployment verification

Review date: 2026-09-06 (UTC).

- The owner verified the public Streamlit deployment using the prompt
  `What is a calm game to play after work?`.
- Retrieval returned five player reviews and GPT-OSS 120B produced a cited answer.
- A screenshot of the successful result is saved at
  `outputs/figures/live_app_gpt_oss.png`.
- The unavailable `llama-3.3-70b-versatile` option was removed after Groq returned
  `404 model_not_found`; the interface now exposes only the verified GPT-OSS model.
# Retrieval benchmark — September 30, 2026

Executed the live retrieval design on 80 versioned questions: 40 development and 40 holdout, with disjoint named-game sets. The run used the pinned original 41,170-review corpus and FAISS index, pinned embedding/reranker revisions and CPU inference. Every question ran through the original 20-candidate stage and reranking of that same pool. Committed per-query records, aggregate manifest and summary show measured game-coverage proxies, incomplete source-anchor recovery and warm latency. No generator request was made; citation correctness, claim support and abstention remain unjudged. Full source review text and credentials are not committed in the new benchmark output.

Eight new dependency-free tests distinguish correct game labels from source-anchor recovery, check two-game coverage, absent evidence slots, unsupported-query missing metrics, valid citation numbers versus factual support, latency interpolation and disjoint benchmark splits. Existing artifact, prompt and diagnostics tests also pass. CI recomputes aggregate proxy metrics from the per-query evidence and checks benchmark/corpus/model metadata without downloading models or calling a provider.

## Benchmark metadata edit — 2026-10-05

Edited the free-text `authoring` description in `evaluation/benchmark.json`. All other fields (80 questions, splits, source anchors, corpus hash and model revisions) were verified unchanged, and `benchmark_sha256` in the baseline manifest was updated to the edited file. The recorded retrieval results are unaffected.

## Retrieval API and regression gate — 2026-10-08

- Added a FastAPI service (`/health`, `/games`, `/search`, `/answer`) that wraps the unchanged `SteamReviewRAG` backend. `retrieval.py`, `prompting.py`, `evaluation.py` and the benchmark runner are byte-identical to the evidence baseline, so `verify_retrieval_evidence.py` still passes.
- 9 API contract tests use a deterministic in-memory backend. They cover input validation, rejection of unknown fields, cache hits, evidence-only mode, out-of-range citation detection, safe provider diagnostics and 503 on backend failure. 4 tests cover the regression gate itself.
- This environment could not reach Hugging Face, so the real-model benchmark rerun, the Docker build and the container queries run in the `Retrieval gate` workflow, not here. Their results are recorded below once that workflow has run.
- `Retrieval gate` run 37812438834 (GitHub Actions, CPU) reran all 80 questions with the real models. Every split/category metric matched the committed baseline exactly. The reranked top-5 reviews were identical for all 80 questions (mean Jaccard 1.000), and the maximum group warm-latency p95 was 1.75 s.
- In the same run the container failed at startup: artifacts written via `NamedTemporaryFile` are mode 0600, so the non-root user could not read them. The Dockerfile now opens read access.
- In run 37813089090 both jobs passed. The container started in 5.0 s and `/health` reported 41,170 reviews and 241 games. `/search` for "A relaxing farming game to play after work" returned Stardew Valley, Stardew Valley and Age of Empires II (Retired), and `/answer` returned evidence in evidence-only mode.
