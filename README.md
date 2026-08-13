# Steam Game Reviews — NLP, DistilBERT & RAG

An applied NLP case study on ~87K Steam game reviews, combining **sentiment classification, semantic search, topic modelling, explainability, and retrieval-augmented generation**.

The project is designed around a practical question: **how can unstructured customer feedback be converted into searchable, measurable product insight?**

## Executive summary

The workflow combines classical NLP methods with transformer-based modelling and a RAG application:

**Raw reviews → preprocessing → embeddings / classification → topic discovery → retrieval → grounded Q&A**

### Key results

| Approach | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF | 89.3% | 0.836 |
| Word2Vec | 85.1% | 0.789 |
| Sentence-BERT | 85.0% | 0.786 |
| **Fine-tuned DistilBERT** | **92.9%** | **0.887** |

For the RAG evaluation, the strongest tested configuration produced **17/18 grounded answers** on the evaluation set.

## What the project demonstrates

- **Customer-feedback analytics:** turning large-scale reviews into structured sentiment and topic signals
- **NLP modelling:** TF-IDF, Word2Vec, Sentence-BERT and fine-tuned DistilBERT
- **Explainable AI:** SHAP-based token-level interpretation of model behaviour
- **Semantic search:** dense embeddings and FAISS retrieval
- **RAG:** retrieval-augmented Q&A over the review corpus
- **Decision support:** making unstructured feedback easier to query and investigate

## Project workflow

| Stage | Output |
|---|---|
| Data preparation | Clean review corpus |
| Text analytics | TF-IDF, word frequencies, EDA |
| Representation | Word2Vec and Sentence-BERT embeddings |
| Classification | DistilBERT sentiment model |
| Explainability | SHAP token importance |
| Topic modelling | LDA topics and prevalence |
| Retrieval | FAISS semantic search |
| Decision layer | Streamlit RAG application |

## Interactive application

`app.py` and `rag_backend.py` provide a Streamlit interface for querying the review corpus through the RAG pipeline.

The application is intended as a demonstration of how unstructured customer feedback can become an **interactive analytics interface**, rather than simply a model benchmark.

## Visual analysis

![DistilBERT confusion matrix](outputs/figures/distilbert_confusion_matrix.png)
![t-SNE of review embeddings by sentiment](outputs/figures/tsne_sentiment.png)
![SHAP global token importance](outputs/figures/shap_global_tokens.png)
![Topic modelling — top words per topic](outputs/figures/lda_topic_words.png)

More figures are available in [`outputs/figures/`](outputs/figures/).

## Repository structure

The notebooks progress from data preparation through modelling and application development:

```text
00  Data preparation
01  Loading & preprocessing
02  Feature engineering & visualization
03  Embeddings & semantic analysis
04  DistilBERT classification & XAI
05  RAG system
06  Topic modelling
```

## Reproduce locally

Install the main dependencies:

```bash
pip install torch faiss-cpu sentence-transformers transformers datasets \
    scikit-learn pandas matplotlib seaborn shap gensim \
    groq python-dotenv langdetect accelerate streamlit pyLDAvis
```

The raw Kaggle dataset and the large FAISS index are intentionally excluded from the repository. A Groq API key is required for the RAG generation layer.

## Tech stack

**Python · PyTorch · Hugging Face Transformers · DistilBERT · Sentence-Transformers · FAISS · SHAP · Gensim · scikit-learn · Streamlit · Groq API**

## Context

Applied NLP portfolio case study developed during an MSc Data Science programme at **The American College of Greece**. The academic context is retained for transparency; the repository is structured around the analytical workflow and its practical use cases.

## Author

**Dimitris Bechrakis**  
Business & Data Analyst | M.Sc. Data Science
