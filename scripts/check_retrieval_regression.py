"""Fail when a fresh retrieval run is worse than the committed baseline.

Used by the retrieval-gate workflow after `scripts/evaluate_retrieval.py` has run the
80-question benchmark with the pinned corpus and model revisions.

    python scripts/check_retrieval_regression.py --run outputs/evaluation/ci-run
"""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
GATED = ["target_game_precision_at_5", "target_game_coverage_at_5"]


def load(run_dir: Path) -> tuple[dict, dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = {r["query_id"]: r for r in json.loads((run_dir / "per_query.json").read_text())}
    return manifest, records


def compare(baseline_dir: Path, run_dir: Path, tolerance: float, min_overlap: float) -> list[str]:
    base_manifest, base_records = load(baseline_dir)
    run_manifest, run_records = load(run_dir)
    failures = []
    for key in ("benchmark_sha256", "corpus_sha256", "model_revisions"):
        if base_manifest[key] != run_manifest[key]:
            failures.append(f"{key} differs from the baseline; the comparison would be meaningless")
    if set(base_records) != set(run_records):
        failures.append("The run does not cover the same benchmark questions")
        return failures

    base_groups = {(g["split"], g["category"]): g for g in base_manifest["results"]}
    print(f"{'split/category':32} {'metric':42} {'baseline':>9} {'run':>9}")
    for group in run_manifest["results"]:
        key = (group["split"], group["category"])
        for stage in ("bi_encoder", "reranked"):
            for metric in GATED:
                name = f"{stage}_{metric}"
                before, after = base_groups[key][name], group[name]
                if before is None or after is None:
                    continue
                flag = "  REGRESSION" if after < before - tolerance else ""
                print(f"{'/'.join(key):32} {name:42} {before:9.3f} {after:9.3f}{flag}")
                if flag:
                    failures.append(f"{'/'.join(key)} {name}: {before:.3f} -> {after:.3f}")

    overlaps = []
    for query_id, record in run_records.items():
        before = {r["review_id"] for r in base_records[query_id]["top_5"]["reranked"]}
        after = {r["review_id"] for r in record["top_5"]["reranked"]}
        overlaps.append(len(before & after) / len(before | after) if before | after else 1.0)
    mean_overlap = sum(overlaps) / len(overlaps)
    changed = sum(o < 1 for o in overlaps)
    print(f"\nReranked top-5 overlap with baseline: mean Jaccard {mean_overlap:.3f}; {changed} of {len(overlaps)} questions changed")
    if mean_overlap < min_overlap:
        failures.append(f"Mean top-5 overlap {mean_overlap:.3f} is below {min_overlap}")

    p95 = [g["warm_total_latency_p95_seconds"] for g in run_manifest["results"]]
    print(f"Warm retrieval latency p95 by group: max {max(p95):.2f}s (informational; runners vary)")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ROOT / "outputs/evaluation/baseline-v1")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Allowed absolute drop per group metric")
    parser.add_argument("--min-overlap", type=float, default=0.8, help="Minimum mean top-5 Jaccard overlap")
    args = parser.parse_args()
    failures = compare(args.baseline, args.run, args.tolerance, args.min_overlap)
    if failures:
        print("\nRetrieval gate FAILED:\n- " + "\n- ".join(failures))
        sys.exit(1)
    print("\nRetrieval gate passed: no group metric fell by more than the tolerance.")


if __name__ == "__main__":
    main()
