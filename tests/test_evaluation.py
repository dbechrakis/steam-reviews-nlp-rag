"""Metric counterexamples and frozen benchmark contracts, no model downloads."""

import json
from pathlib import Path
import unittest

from steam_review_rag.evaluation import citation_metrics, percentile, retrieval_metrics


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def test_game_coverage_does_not_imply_source_anchor_recovery(self):
        query = {"category": "game_topic", "target_games": ["A"], "anchor_review_ids": ["7"]}
        rows = [{"game_name": "A", "raw_row_id": i} for i in range(1, 6)]
        result = retrieval_metrics(query, rows)
        self.assertEqual(result["target_game_precision_at_5"], 1)
        self.assertEqual(result["anchor_hit_at_5"], 0)
        self.assertEqual(result["anchor_reciprocal_rank"], 0)

    def test_anchor_rank_is_calculated_across_full_candidate_pool(self):
        query = {"category": "game_topic", "target_games": ["A"], "anchor_review_ids": ["6"]}
        rows = [{"game_name": "A", "raw_row_id": i} for i in range(1, 7)]
        result = retrieval_metrics(query, rows)
        self.assertEqual(result["anchor_hit_at_5"], 0)
        self.assertAlmostEqual(result["anchor_reciprocal_rank"], 1 / 6)

    def test_comparison_requires_both_games(self):
        query = {"category": "comparison", "target_games": ["A", "B"], "anchor_review_ids": []}
        result = retrieval_metrics(query, [{"game_name": "A", "raw_row_id": 1}] * 5)
        self.assertEqual(result["target_game_precision_at_5"], 1)
        self.assertEqual(result["target_game_coverage_at_5"], 0.5)
        self.assertIsNone(result["anchor_hit_at_5"])

    def test_missing_candidates_are_not_dropped_from_precision_denominator(self):
        query = {"category": "named_game", "target_games": ["A"]}
        result = retrieval_metrics(query, [{"game_name": "A", "raw_row_id": 1}])
        self.assertEqual(result["target_game_precision_at_5"], 0.2)

    def test_unsupported_queries_do_not_receive_factuality_proxy(self):
        result = retrieval_metrics({"category": "unsupported", "target_games": []}, [])
        self.assertTrue(all(v is None for v in result.values()))

    def test_invalid_and_absent_citations_are_distinct_from_unmeasured(self):
        self.assertFalse(citation_metrics("Claim [0] [6]", 5)["citation_numbering_valid"])
        self.assertFalse(citation_metrics("Claim without citations", 5)["citation_numbering_valid"])
        self.assertIsNone(citation_metrics(None, 5)["citation_numbering_valid"])
        self.assertTrue(citation_metrics("A fabricated claim [1]", 5)["citation_numbering_valid"])

    def test_latency_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.95), 3.85)
        self.assertIsNone(percentile([], 0.5))

    def test_benchmark_has_80_unique_queries_and_disjoint_game_splits(self):
        benchmark = json.loads((ROOT / "evaluation/benchmark.json").read_text())
        queries = benchmark["queries"]
        self.assertEqual(len(queries), 80)
        self.assertEqual(len({q["query_id"] for q in queries}), 80)
        game_sets = {}
        for split in ("development", "holdout"):
            selected = [q for q in queries if q["split"] == split]
            self.assertEqual(len(selected), 40)
            game_sets[split] = {g for q in selected for g in q["target_games"]}
        self.assertFalse(game_sets["development"] & game_sets["holdout"])
        self.assertEqual(sum(q["category"] == "game_topic" for q in queries), 40)
        for query in queries:
            self.assertEqual(query["anchor_review_ids"], [a["review_id"] for a in query["anchors"]])
            self.assertTrue(query["question"].strip())


if __name__ == "__main__":
    unittest.main()
