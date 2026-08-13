# Steam Reviews — NLP, Explainability & RAG

An end-to-end **customer-feedback analytics** project using ~87K Steam game reviews to classify sentiment, discover themes, perform semantic search, explain model predictions, and build a retrieval-augmented Q&A system.

The project demonstrates how unstructured customer feedback can be transformed into **measurable signals and an interactive decision-support workflow**.

## Business questions

- What is the overall sentiment of player feedback?
- Which themes appear repeatedly across reviews?
- Can similar reviews be retrieved semantically rather than by keyword alone?
- Which features drive sentiment predictions?
- Can a grounded RAG system answer questions using the review corpus as evidence?

## Key results

### Sentiment classification

| Model / representation | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF | 89.3% | 0.836 |
| Word2Vec | 85.1% | 0.789 |
| Sentence-BERT | 85.0% | 0.786 |
| **Fine-tuned DistilBERT** | **92.9%** | **0.887** |

### RAG evaluation

The strongest tested configuration produced **17/18 grounded answers** on the evaluation set.

## Analytics workflow

```text
Raw Reviews
    ↓
Cleaning & Language Detection
    ↓
EDA / TF-IDF / Embeddings
    ↓
Semantic Search + Topic Modelling
    ↓
DistilBERT Sentiment Classification
    ↓
SHAP Explainability
    ↓
FAISS Retrieval
    ↓
RAG Q&A Application
```

## Project components

| Component | Purpose |
|---|---|
| Data preparation | Cleaning, language detection and normalization |
| Text analytics | TF-IDF, word frequencies and exploratory analysis |
| Embeddings | Word2Vec and Sentence-BERT semantic representations |
| Classification | DistilBERT sentiment model and baselines |
| Explainability | SHAP analysis of model behaviour |
| Topic modelling | LDA-based discovery of recurring themes |
| Retrieval | FAISS semantic search over review embeddings |
| RAG | Grounded Q&A over the review corpus |
| Application | Streamlit interface for interactive exploration |

## Visual analysis

The repository includes confusion-matrix, embedding, SHAP and topic-modelling outputs under `outputs/figures/`.

## Tech stack

**Python · PyTorch · Hugging Face Transformers · DistilBERT · Sentence-Transformers · FAISS · SHAP · Gensim · scikit-learn · Streamlit**

## Reproducibility

The notebooks include their generated outputs for review. The raw Kaggle dataset and the large FAISS index are intentionally excluded from the repository. A Groq API key is required for the RAG generation layer.

Never commit API credentials or `.env` files.

## Context

Applied NLP portfolio case study developed during an MSc Data Science programme at **The American College of Greece**. The academic context is retained for transparency; the repository is presented around the analytical problem, methodology, results and application.

## Author

**Dimitris Bechrakis**  
Business & Data Analyst | M.Sc. Data Science
