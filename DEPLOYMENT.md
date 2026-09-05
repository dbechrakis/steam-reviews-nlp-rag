# Deploy the Steam review explorer

## Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Create app → Deploy a public app from GitHub.
3. Repository: `dbechrakis/steam-reviews-nlp-rag`
4. Branch: `main`
5. Main file path: `deploy/streamlit_app.py`
6. Advanced settings → Python **3.12**. Add this in Secrets, replacing the placeholder:

```toml
GROQ_API_KEY = "YOUR_OWN_GROQ_KEY"
```

7. Deploy. Initial installation and model downloads can take several minutes.

Do not commit the real key or share screenshots of it. Without a key, the app
still retrieves and displays reviews. A configured key is not proof that the
provider accepted it: test a question and verify an AI answer plus evidence.

The entrypoint directory has its own pinned, CPU-only requirements. The root
requirements remain the wider notebook environment. Do not select root `app.py`
for Community Cloud: select `deploy/streamlit_app.py` to use the lean dependency set.

## What downloads automatically

`artifact_loader.py` downloads ~110 MB from the original team deployment:
https://huggingface.co/spaces/Nebuchedeser/steam-game-review-explorer

Revision: `c9a4d43b9422edde3fe85b20cc8f4645e258ac02`.
The corpus, configuration and FAISS index are checked against fixed sizes and
hashes before use. This preserves matching row/vector order and avoids fetching
an unreviewed latest export. SentenceTransformer and CrossEncoder weights also
download on first search. No transformer training is needed.

The source Space is owned by teammate Nebuchedeser. This portfolio deployment
reuses the team's exported retrieval artifacts and retains the project's
existing authorship credit. It does not claim sole authorship or relicense the
source corpus. The recorded classification scores refer to their documented
test set, not the retrieval corpus.

Pausing the source app does not remove its public repository files. However,
deletion or access changes to the source repository would prevent a fresh
download. This deployment is therefore not an independent data archive.

## Runtime behaviour and limitations

- Each question is retrieved independently; the visible conversation is not passed as model memory.
- No API key is accepted from visitors or displayed to them.
- Questions and selected review excerpts are sent to Groq for generation.
- Generation failures retain retrieved evidence; retrieval failures display a friendly error.
- There is a five-second per-session cooldown and a 600-character question limit.
  These are modest demo controls, not global abuse protection. Use provider-side
  quotas/spend limits before enabling paid usage.
- Free/shared hosting may sleep or hit resource limits. A successful local test
  does not establish Community Cloud uptime or memory headroom.
- Generated citations can still be wrong; visitors can inspect the actual reviews.

## Local check

```bash
python -m pip install -r deploy/requirements.txt
python artifact_loader.py
python -m streamlit run deploy/streamlit_app.py
```

After deployment, ask `What is a calm game to play after work?`, expand the
evidence and verify that the response and cited reviews are shown. If generation
is unavailable, check the key and provider quota in the owner dashboard.
