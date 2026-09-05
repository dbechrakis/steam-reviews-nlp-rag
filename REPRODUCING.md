# Reproducing the review analytics

Use Python 3.11 or 3.12 in a virtual environment and install `requirements.txt` plus Jupyter.
The source CSVs are not included. Preserve the original Kaggle dataset/version; do not substitute a different corpus and claim reproduction of the recorded scores.

Required input files in `data/`:
- `steam_game_reviews.csv`
- `games_description.csv`
- `games_ranking.csv`

Run notebook 00 first for the canonical modelling sample and game-grouped splits. Inspect each notebook's inputs before running 01–06 in order; 01 is an exploratory sample, not the canonical model split. Notebook 03 exports embeddings and row-aligned metadata; 04 trains the classifier; 05 exports `outputs/rag/{rag_config.json,rag_corpus.csv,faiss_index.bin}`. Then run `streamlit run app.py`.

For live generation, provide your own `GROQ_API_KEY` through the environment or an untracked `.env`. Verify generator availability with the provider before execution. Training requires model downloads and substantial compute.

The September 2026 review checked saved result tables and evaluation code. It did not retrain the transformer or regenerate answers. The name-presence metric is a weak proxy: claim-level support requires a separately labelled evaluation set, including unanswerable questions.
