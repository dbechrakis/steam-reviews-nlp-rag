"""REST service over the two-stage review retrieval and optional grounded answers.

Run locally with ``uvicorn steam_review_rag.api:app``. The service wraps the same
``SteamReviewRAG`` backend as the Streamlit app, so retrieval and prompting cannot drift
between the two. Heavy model imports happen at startup, not at import, which keeps the
contract tests free of torch and model downloads.
"""

from collections import OrderedDict
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Literal, Protocol

from fastapi import FastAPI, HTTPException, Request
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from steam_review_rag import __version__
from steam_review_rag.diagnostics import generation_diagnostic
from steam_review_rag.evaluation import citation_metrics


PROJECT_DIR = Path(os.environ.get("STEAM_RAG_PROJECT_DIR", Path(__file__).resolve().parents[2]))
ANSWER_MODEL_ID = "openai/gpt-oss-120b"
MAX_EVIDENCE = 8
CACHE_SIZE = 512


class Backend(Protocol):
    corpus: pd.DataFrame
    device: str

    def retrieve_and_rerank(self, question: str, evidence_count: int | None = None) -> pd.DataFrame: ...

    def generate_answer(self, question: str, evidence: pd.DataFrame, api_key: str, model_name: str) -> str: ...


def load_default_backend() -> Backend:
    """Verify the pinned artifacts, load the index and warm both models."""
    from steam_review_rag.artifacts import ensure_artifacts
    from steam_review_rag.retrieval import SteamReviewRAG

    ensure_artifacts(PROJECT_DIR)
    backend = SteamReviewRAG.load(PROJECT_DIR)
    # Serve the exact model revisions the retrieval benchmark (and its CI gate) measured.
    benchmark = PROJECT_DIR / "evaluation" / "benchmark.json"
    if benchmark.exists():
        backend.config.update(json.loads(benchmark.read_text())["model_revisions"])
    backend.retrieve_and_rerank("warm up the encoder and reranker", 1)
    return backend


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=600, examples=["A calm game to play after work"])
    evidence_count: int = Field(default=5, ge=1, le=MAX_EVIDENCE)

    @field_validator("question")
    @classmethod
    def normalise(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("Question must contain text")
        return value


class Evidence(BaseModel):
    rank: int
    review_id: str
    game_name: str
    recommendation: str
    review: str
    bi_score: float = Field(description="Bi-encoder cosine similarity (FAISS stage)")
    rerank_score: float = Field(description="Cross-encoder relevance logit (reranking stage)")


class SearchResponse(BaseModel):
    question: str
    evidence: list[Evidence]
    cached: bool
    retrieval_ms: float


class CitationCheck(BaseModel):
    citation_count: int | None
    invalid_citation_count: int | None
    citation_numbering_valid: bool | None


class AnswerResponse(SearchResponse):
    answer: str | None
    generation_status: Literal["generated", "not_configured", "failed"]
    diagnostic: str | None = None
    citation_check: CitationCheck = Field(
        description="Checks that bracket citations point at supplied evidence; not that claims are supported"
    )
    generation_ms: float | None = None


class RetrievalCache:
    """Bounded LRU of reranked evidence; safe because the corpus and models are pinned."""

    def __init__(self, size: int = CACHE_SIZE):
        self.size = size
        self.items: OrderedDict[tuple[str, int], pd.DataFrame] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, key: tuple[str, int]) -> pd.DataFrame | None:
        with self.lock:
            if key in self.items:
                self.items.move_to_end(key)
                self.hits += 1
                return self.items[key]
            self.misses += 1
            return None

    def put(self, key: tuple[str, int], value: pd.DataFrame) -> None:
        with self.lock:
            self.items[key] = value
            self.items.move_to_end(key)
            while len(self.items) > self.size:
                self.items.popitem(last=False)


def create_app(backend_factory=load_default_backend) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.perf_counter()
        app.state.backend = backend_factory()
        app.state.startup_seconds = time.perf_counter() - started
        app.state.cache = RetrievalCache()
        # Torch inference is CPU-bound; one request at a time keeps latency predictable.
        app.state.inference_lock = threading.Lock()
        yield

    app = FastAPI(
        title="Steam Review Retrieval API",
        version=__version__,
        description=(
            "Semantic search over 41,170 Steam reviews (FAISS + cross-encoder reranking) and "
            "optional answers grounded in the returned reviews. Citations are checked for "
            "numbering only; generated claims can still be unsupported."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def timing_header(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    def search(request: Request, payload: Question) -> tuple[pd.DataFrame, bool, float]:
        key = (payload.question.casefold(), payload.evidence_count)
        cache: RetrievalCache = request.app.state.cache
        started = time.perf_counter()
        evidence = cache.get(key)
        if evidence is not None:
            return evidence, True, (time.perf_counter() - started) * 1000
        try:
            with request.app.state.inference_lock:
                evidence = request.app.state.backend.retrieve_and_rerank(
                    payload.question, payload.evidence_count
                )
        except Exception as error:
            logging.exception("Retrieval failed")
            raise HTTPException(status_code=503, detail="Review search is temporarily unavailable") from error
        cache.put(key, evidence)
        return evidence, False, (time.perf_counter() - started) * 1000

    def evidence_items(evidence: pd.DataFrame) -> list[Evidence]:
        return [
            Evidence(
                rank=position,
                review_id=str(row.get("raw_row_id", row.get("corpus_row_id"))),
                game_name=str(row["game_name"]),
                recommendation=str(row["recommendation"]),
                review=str(row["review"]),
                bi_score=float(row["bi_score"]),
                rerank_score=float(row["rerank_score"]),
            )
            for position, row in enumerate(evidence.to_dict("records"), start=1)
        ]

    @app.get("/health")
    def health(request: Request) -> dict:
        backend = request.app.state.backend
        cache: RetrievalCache = request.app.state.cache
        return {
            "status": "ok",
            "corpus_reviews": len(backend.corpus),
            "games": int(backend.corpus["game_name"].nunique()),
            "device": backend.device,
            "startup_seconds": round(request.app.state.startup_seconds, 2),
            "answers_enabled": bool(os.getenv("GROQ_API_KEY")),
            "cache": {"entries": len(cache.items), "hits": cache.hits, "misses": cache.misses},
        }

    @app.get("/games")
    def games(request: Request) -> list[dict]:
        counts = request.app.state.backend.corpus["game_name"].value_counts()
        return [{"game_name": name, "reviews": int(count)} for name, count in counts.items()]

    @app.post("/search", response_model=SearchResponse)
    def search_reviews(payload: Question, request: Request) -> SearchResponse:
        evidence, cached, elapsed = search(request, payload)
        return SearchResponse(
            question=payload.question, evidence=evidence_items(evidence), cached=cached, retrieval_ms=elapsed
        )

    @app.post("/answer", response_model=AnswerResponse)
    def answer(payload: Question, request: Request) -> AnswerResponse:
        evidence, cached, elapsed = search(request, payload)
        key = os.getenv("GROQ_API_KEY")
        text, status, diagnostic, generation_ms = None, "not_configured", None, None
        if key:
            started = time.perf_counter()
            try:
                text = request.app.state.backend.generate_answer(
                    payload.question, evidence, key.strip(), ANSWER_MODEL_ID
                )
                status = "generated"
            except Exception as error:  # noqa: BLE001 - evidence stays available on any provider failure
                status, diagnostic = "failed", generation_diagnostic(error, ANSWER_MODEL_ID)
            generation_ms = (time.perf_counter() - started) * 1000
        return AnswerResponse(
            question=payload.question,
            evidence=evidence_items(evidence),
            cached=cached,
            retrieval_ms=elapsed,
            answer=text,
            generation_status=status,
            diagnostic=diagnostic,
            citation_check=CitationCheck(**citation_metrics(text, len(evidence))),
            generation_ms=generation_ms,
        )

    return app


app = create_app()
