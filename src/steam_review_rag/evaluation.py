"""Dependency-free metrics with explicit boundaries between proxies and judgments."""

import hashlib
import math
import re
from statistics import mean


def review_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def retrieval_metrics(query: dict, rows: list[dict], k: int = 5) -> dict:
    """Measure target-game coverage and known-anchor recovery, not factuality.

    Source anchors are incomplete relevance labels: other reviews may also answer.
    A matching game name does not establish topical support for a question.
    Unsupported questions require answer-level adjudication and receive no proxy.
    """
    if k < 1:
        raise ValueError("k must be positive")
    top = rows[:k]
    targets = set(query["target_games"])
    anchors = set(query.get("anchor_review_ids", []))
    if query["category"] == "unsupported":
        return {"target_game_precision_at_5": None, "target_game_coverage_at_5": None,
                "anchor_hit_at_5": None, "anchor_reciprocal_rank": None}
    present = {row["game_name"] for row in top}
    ranks = [i for i, row in enumerate(rows, start=1) if str(row["raw_row_id"]) in anchors]
    return {
        "target_game_precision_at_5": sum(row["game_name"] in targets for row in top) / k,
        "target_game_coverage_at_5": len(present & targets) / len(targets) if targets else None,
        "anchor_hit_at_5": int(any(rank <= k for rank in ranks)) if anchors else None,
        "anchor_reciprocal_rank": 1 / min(ranks) if ranks else (0 if anchors else None),
    }


def citation_metrics(answer: str | None, evidence_count: int) -> dict:
    """Check citation numbering only; correctness of a cited claim needs a reviewer."""
    if answer is None:
        return {"citation_count": None, "invalid_citation_count": None,
                "citation_numbering_valid": None}
    citations = [int(number) for number in re.findall(r"\[(\d+)\]", answer)]
    invalid = sum(not 1 <= number <= evidence_count for number in citations)
    return {"citation_count": len(citations), "invalid_citation_count": invalid,
            "citation_numbering_valid": bool(citations) and invalid == 0}


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0 <= probability <= 1:
        raise ValueError("Probability must lie between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate_results(results: list[dict]) -> list[dict]:
    """Macro-average questions within split/category; preserve missing judgments."""
    groups = sorted({(r["split"], r["category"]) for r in results})
    summary = []
    for split, category in groups:
        selected = [r for r in results if r["split"] == split and r["category"] == category]
        row = {"split": split, "category": category, "queries": len(selected)}
        for stage in ("bi_encoder", "reranked"):
            for metric in ("target_game_precision_at_5", "target_game_coverage_at_5",
                           "anchor_hit_at_5", "anchor_reciprocal_rank"):
                values = [r[stage][metric] for r in selected if r[stage][metric] is not None]
                row[stage + "_" + metric] = mean(values) if values else None
        row["warm_total_latency_p50_seconds"] = percentile(
            [r["latency_seconds"]["total_retrieval"] for r in selected], 0.5)
        row["warm_total_latency_p95_seconds"] = percentile(
            [r["latency_seconds"]["total_retrieval"] for r in selected], 0.95)
        row["generated_answers"] = sum(r.get("generation_status") == "generated" for r in selected)
        summary.append(row)
    return summary
