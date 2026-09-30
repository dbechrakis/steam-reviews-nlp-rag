"""Verify recorded proxy metrics and provenance without inference or provider calls."""

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from steam_review_rag.artifacts import FILES, REVISION  # noqa: E402
from steam_review_rag.evaluation import aggregate_results, citation_metrics, retrieval_metrics  # noqa: E402


def verify() -> None:
    benchmark_path = ROOT / "evaluation/benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    run_dir = ROOT / "outputs/evaluation/baseline-v1"
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = json.loads((run_dir / "per_query.json").read_text())
    assert manifest["benchmark_sha256"] == hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    assert manifest["corpus_sha256"] == benchmark["corpus_sha256"] == FILES["rag_corpus.csv"][1]
    assert manifest["artifact_revision"] == REVISION
    assert manifest["model_revisions"] == benchmark["model_revisions"]
    assert len(records) == manifest["queries"] == 80
    assert not manifest["generation_requested"]
    queries = {q["query_id"]: q for q in benchmark["queries"]}
    assert len({r["query_id"] for r in records}) == len(queries)
    for path, digest in manifest["code_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    for record in records:
        query = queries[record["query_id"]]
        for key in ("split", "category", "question"):
            assert record[key] == query[key]
        pool = record["rankings"]
        for stage in ("bi_encoder", "reranked"):
            assert len(pool[stage]) == 20
            assert len({r["raw_row_id"] for r in pool[stage]}) == 20
            assert record[stage] == retrieval_metrics(query, pool[stage]), record["query_id"]
            assert [(r["review_id"], r["game_name"]) for r in record["top_5"][stage]] == [
                (r["raw_row_id"], r["game_name"]) for r in pool[stage][:5]]
        assert {r["raw_row_id"] for r in pool["bi_encoder"]} == {r["raw_row_id"] for r in pool["reranked"]}
        assert record["generation_status"] == "not_requested"
        assert record["answer"] is None and record["claim_support_status"] == "unjudged"
        assert record["citation_numbering"] == citation_metrics(None, 5)
        assert record["latency_seconds"]["total_retrieval"] >= 0
    assert manifest["results"] == aggregate_results(records)
    print("80-query retrieval evidence, aggregate arithmetic and provenance verified.")


if __name__ == "__main__":
    verify()
