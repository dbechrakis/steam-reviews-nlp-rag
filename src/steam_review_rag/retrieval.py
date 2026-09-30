"""Two-stage retrieval and grounded-answer utilities for Steam reviews."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from steam_review_rag.prompting import build_grounded_prompt


def preferred_device() -> str:
    """Choose the best locally available inference device."""
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
        """Load and validate a matching configuration, corpus, and FAISS index."""
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
                f"RAG index has {index.ntotal:,} vectors but the corpus has "
                f"{len(corpus):,} rows. Use matching artifacts from the same notebook run."
            )

        required_columns = {"game_name", "review", "recommendation"}
        if not required_columns.issubset(corpus.columns) or corpus.empty:
            raise ValueError("Invalid review corpus schema")
        if corpus[list(required_columns)].isna().any().any():
            raise ValueError("Missing review evidence fields")
        return cls(project_dir, config, index, corpus, preferred_device())

    def _load_models(self) -> None:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                self.config["embedding_model"], device=self.device,
                revision=self.config.get("embedding_revision"),
            )
        if self.reranker is None:
            self.reranker = CrossEncoder(
                self.config["reranker_model"], device=self.device,
                revision=self.config.get("reranker_revision"),
            )

    def retrieve_and_rerank(
        self,
        question: str,
        evidence_count: int | None = None,
    ) -> pd.DataFrame:
        """Retrieve semantic candidates and rerank them for the user question."""
        candidates = self.retrieve_candidates(question)
        return self.rerank_candidates(question, candidates, evidence_count)

    def retrieve_candidates(self, question: str) -> pd.DataFrame:
        """Expose the unchanged bi-encoder stage and stable corpus row identifiers."""
        question = question.strip()
        if not question:
            return self.corpus.iloc[0:0].copy()
        self._load_models()
        assert self.embedding_model is not None
        retrieve_k = min(int(self.config.get("retrieve_k", 20)), len(self.corpus))
        if retrieve_k < 1:
            raise ValueError("retrieve_k must be positive")
        query_vector = self.embedding_model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")
        scores, row_ids = self.index.search(query_vector, retrieve_k)
        if (row_ids[0] < 0).any() or (row_ids[0] >= len(self.corpus)).any():
            raise ValueError("FAISS returned an invalid corpus row identifier")
        candidates = self.corpus.iloc[row_ids[0]].copy().reset_index(drop=True)
        candidates.insert(0, "corpus_row_id", row_ids[0])
        candidates.insert(1, "bi_score", scores[0])
        return candidates

    def rerank_candidates(
        self, question: str, candidates: pd.DataFrame, evidence_count: int | None = None,
    ) -> pd.DataFrame:
        """Rerank the same candidate pool so evaluation isolates the second stage."""
        if candidates.empty:
            return candidates.copy()
        self._load_models()
        assert self.reranker is not None
        final_k = evidence_count or int(self.config.get("final_k", 5))
        final_k = max(1, min(final_k, len(candidates)))
        candidates = candidates.copy()
        pairs = list(zip([question.strip()] * len(candidates), candidates["review"].tolist()))
        candidates["rerank_score"] = self.reranker.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )
        return (
            candidates.sort_values("rerank_score", ascending=False)
            .head(final_k)
            .reset_index(drop=True)
        )

    def build_prompt(self, question: str, evidence: pd.DataFrame) -> str:
        """Build the grounded generation prompt from retrieved evidence."""
        return build_grounded_prompt(question, evidence.to_dict("records"))

    def generate_answer(
        self,
        question: str,
        evidence: pd.DataFrame,
        api_key: str,
        model_name: str,
    ) -> str:
        """Generate one evidence-constrained answer through Groq."""
        from groq import Groq

        client = Groq(api_key=api_key, timeout=30, max_retries=1)
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
