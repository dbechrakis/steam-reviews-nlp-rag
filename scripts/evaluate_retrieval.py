"""Run the deployed retrieval stages on a frozen, corpus-backed query benchmark."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steam_review_rag.evaluation import (  # noqa: E402
    aggregate_results, citation_metrics, retrieval_metrics, review_hash,
)


def run(output: Path, generate: bool = False) -> dict:
    # Imports are lazy so CI can validate benchmark contracts without model downloads.
    import importlib.metadata
    from steam_review_rag.artifacts import REVISION, ensure_artifacts
    from steam_review_rag.diagnostics import generation_diagnostic
    from steam_review_rag.retrieval import SteamReviewRAG

    benchmark_path = ROOT / "evaluation" / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    ensure_artifacts(ROOT)
    corpus_path = ROOT / "outputs" / "rag" / "rag_corpus.csv"
    corpus_sha = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if corpus_sha != benchmark["corpus_sha256"]:
        raise ValueError("Benchmark and runtime corpus are different snapshots")
    if generate and not os.getenv("GROQ_API_KEY"):
        raise ValueError("--generate requires an explicitly configured GROQ_API_KEY")
    backend = SteamReviewRAG.load(ROOT)
    backend.config.update(benchmark["model_revisions"])
    if backend.corpus.raw_row_id.duplicated().any():
        raise ValueError("Source review IDs are not unique")
    by_id = backend.corpus.assign(raw_row_id=backend.corpus.raw_row_id.astype(str)).set_index("raw_row_id")
    for query in benchmark["queries"]:
        for anchor in query.get("anchors", []):
            if review_hash(str(by_id.loc[anchor["review_id"], "review"])) != anchor["review_sha256"]:
                raise ValueError("Anchor review hash mismatch")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    backend._load_models()
    startup_seconds = time.perf_counter() - start
    # Exclude the first inference warmup from query latency; report startup separately.
    backend.retrieve_and_rerank("Find a game with interesting puzzles.")
    results = []
    review_pool = []
    for query in benchmark["queries"]:
        start = time.perf_counter()
        candidates = backend.retrieve_candidates(query["question"])
        retrieval_seconds = time.perf_counter() - start
        start = time.perf_counter()
        ranked = backend.rerank_candidates(query["question"], candidates, len(candidates))
        rerank_seconds = time.perf_counter() - start
        candidates_rows = candidates.to_dict("records")
        ranked_rows = ranked.to_dict("records")
        evidence = ranked.head(5)
        answer = None
        status = "not_requested"
        generation_seconds = None
        diagnostic = None
        if generate:
            start = time.perf_counter()
            try:
                answer = backend.generate_answer(query["question"], evidence,
                                                 os.environ["GROQ_API_KEY"], "openai/gpt-oss-120b")
                status = "generated"
            except Exception as error:
                status = "failed"
                diagnostic = generation_diagnostic(error, "openai/gpt-oss-120b")
            generation_seconds = time.perf_counter() - start
        result = {
            "query_id": query["query_id"], "split": query["split"], "category": query["category"],
            "question": query["question"],
            "bi_encoder": retrieval_metrics(query, candidates_rows),
            "reranked": retrieval_metrics(query, ranked_rows),
            "latency_seconds": {"bi_encoder": retrieval_seconds, "reranker": rerank_seconds,
                                "total_retrieval": retrieval_seconds + rerank_seconds,
                                "generation": generation_seconds},
            "generation_status": status, "generation_diagnostic": diagnostic,
            "answer": answer, "citation_numbering": citation_metrics(answer, len(evidence)),
            "claim_support_status": "unjudged",
            "rankings": {stage: [{"raw_row_id": str(r["raw_row_id"]), "game_name": r["game_name"]}
                                 for r in rows]
                         for stage, rows in [("bi_encoder", candidates_rows), ("reranked", ranked_rows)]},
            "top_5": {stage: [{"review_id": str(r["raw_row_id"]), "game_name": r["game_name"],
                               "review_sha256": review_hash(r["review"])} for r in rows[:5]]
                      for stage, rows in [("bi_encoder", candidates_rows), ("reranked", ranked_rows)]},
        }
        results.append(result)
        pooled = {}
        for stage, rows in [("bi_encoder", candidates_rows), ("reranked", ranked_rows)]:
            for rank, row in enumerate(rows[:5], start=1):
                review_id = str(row["raw_row_id"])
                item = pooled.setdefault(review_id, {
                    "query_id": query["query_id"], "question": query["question"],
                    "review_id": review_id, "game_name": row["game_name"],
                    "review_sha256": review_hash(row["review"]),
                    "bi_encoder_rank": "", "reranked_rank": "",
                    "relevance_grade": "", "reviewer": "", "rationale": "",
                })
                item[stage + "_rank"] = rank
        review_pool.extend(pooled.values())
        # Preserve completed questions if interrupted; overwrite only this run directory.
        (output / "per_query.json").write_text(json.dumps(results, indent=2) + "\n")
        print(query["query_id"], status, flush=True)
    with (output / "retrieval_judgments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_pool[0]))
        writer.writeheader()
        writer.writerows(review_pool)
    with (output / "answer_judgments.csv").open("w", newline="") as handle:
        fields = ["query_id", "claim_id", "claim_text", "citation_ids", "support_status",
                  "abstention_appropriate", "reviewer", "rationale"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"query_id": q["query_id"]} for q in benchmark["queries"])
    manifest = {
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "corpus_sha256": corpus_sha, "artifact_revision": REVISION,
        "configuration_sha256": hashlib.sha256((ROOT / "outputs/rag/rag_config.json").read_bytes()).hexdigest(),
        "queries": len(results), "device": backend.device, "python": platform.python_version(),
        "model_startup_seconds": startup_seconds,
        "generation_requested": generate, "generator": "openai/gpt-oss-120b" if generate else None,
        "package_versions": {p: importlib.metadata.version(p) for p in
                             ("torch", "sentence-transformers", "transformers", "faiss-cpu", "pandas")},
        "model_identifiers": {"embedding": backend.config["embedding_model"],
                              "reranker": backend.config["reranker_model"]},
        "model_revisions": benchmark["model_revisions"],
        "code_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                        ("scripts/evaluate_retrieval.py", "src/steam_review_rag/evaluation.py",
                         "src/steam_review_rag/retrieval.py", "src/steam_review_rag/prompting.py")},
        "runtime_note": "Benchmark pins resolved model revisions; the live app retains its existing model-ID defaults.",
        "results": aggregate_results(results),
        "claim_support_status": "unjudged",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation/baseline-v1")
    parser.add_argument("--generate", action="store_true", help="Call Groq; never enabled implicitly")
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.generate), indent=2))
