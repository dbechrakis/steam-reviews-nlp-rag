# Validation record

Review date: 2026-09-05 (UTC).

Checked saved CSV arithmetic, sample sizes and evaluation implementation; no retraining or live generation.

Changes were prepared with AI assistance and should be understood and reviewed by the repository owner. Historical records are not labelled as freshly reproduced results.

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
