"""Retrieval and grounded-answer utilities for the Steam review app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer


def preferred_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SteamReviewRAG:
    project_dir: Path
    config: dict
    index: faiss.Index
    corpus: pd.DataFrame
    device: str
    embedding_model: SentenceTransformer | None = None
    reranker: CrossEncoder | None = None

    @classmethod
    def load(cls, project_dir: Path) -> "SteamReviewRAG":
        rag_dir = project_dir / "outputs" / "rag"
        required = {
            "configuration": rag_dir / "rag_config.json",
            "corpus": rag_dir / "rag_corpus.csv",
            "FAISS index": rag_dir / "faiss_index.bin",
        }
        missing = [label for label, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing RAG artifact(s): {', '.join(missing)}")

        config = json.loads(required["configuration"].read_text())
        corpus = pd.read_csv(required["corpus"], low_memory=False)
        index = faiss.read_index(str(required["FAISS index"]))
        if index.ntotal != len(corpus):
            raise ValueError(
                f"RAG index has {index.ntotal:,} vectors but the corpus has {len(corpus):,} rows. "
                "Use matching artifacts from the same notebook run."
            )
        return cls(project_dir, config, index, corpus, preferred_device())

    def _load_models(self) -> None:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                self.config["embedding_model"], device=self.device
            )
        if self.reranker is None:
            self.reranker = CrossEncoder(self.config["reranker_model"], device=self.device)

    def retrieve_and_rerank(self, question: str, evidence_count: int | None = None) -> pd.DataFrame:
        question = question.strip()
        if not question:
            return self.corpus.iloc[0:0].copy()

        self._load_models()
        assert self.embedding_model is not None and self.reranker is not None

        retrieve_k = min(int(self.config.get("retrieve_k", 20)), len(self.corpus))
        final_k = evidence_count or int(self.config.get("final_k", 5))
        final_k = max(1, min(final_k, retrieve_k))

        query_vector = self.embedding_model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, row_ids = self.index.search(query_vector, retrieve_k)
        candidates = self.corpus.iloc[row_ids[0]].copy().reset_index(drop=True)
        candidates.insert(0, "bi_score", scores[0])

        pairs = list(zip([question] * len(candidates), candidates["review"].tolist()))
        candidates["rerank_score"] = self.reranker.predict(
            pairs, batch_size=32, show_progress_bar=False
        )
        return candidates.sort_values("rerank_score", ascending=False).head(final_k).reset_index(drop=True)

    def build_prompt(self, question: str, evidence: pd.DataFrame) -> str:
        blocks = []
        for number, row in evidence.reset_index(drop=True).iterrows():
            review = " ".join(str(row["review"]).split())[:900]
            blocks.append(
                f"[{number + 1}] Game: {row['game_name']}\n"
                f"Player verdict: {row['recommendation']}\n"
                f"Review evidence: {review}"
            )
        context = "\n\n".join(blocks)
        return (
            f"Player-review evidence:\n\n{context}\n\nQuestion: {question}\n\n"
            "Answer only from the evidence above."
        )

    def generate_answer(self, question: str, evidence: pd.DataFrame, api_key: str, model_name: str) -> str:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": self.config["system_prompt"]},
                {"role": "user", "content": self.build_prompt(question, evidence)},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content
