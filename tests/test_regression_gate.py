import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_retrieval_regression import compare  # noqa: E402

BASELINE = ROOT / "outputs/evaluation/baseline-v1"


def write_run(directory: Path, manifest: dict, records: list) -> Path:
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "per_query.json").write_text(json.dumps(records))
    return directory


class RegressionGateTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((BASELINE / "manifest.json").read_text())
        self.records = json.loads((BASELINE / "per_query.json").read_text())

    def test_identical_run_passes(self):
        self.assertEqual(compare(BASELINE, BASELINE, 0.02, 0.8), [])

    def test_metric_drop_fails(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["results"][0]["reranked_target_game_coverage_at_5"] -= 0.2
        with tempfile.TemporaryDirectory() as tmp:
            failures = compare(BASELINE, write_run(Path(tmp), manifest, self.records), 0.02, 0.8)
        self.assertEqual(len(failures), 1)
        self.assertIn("reranked_target_game_coverage_at_5", failures[0])

    def test_changed_rankings_fail_overlap(self):
        records = copy.deepcopy(self.records)
        for record in records:
            for item in record["top_5"]["reranked"]:
                item["review_id"] = "other-" + item["review_id"]
        with tempfile.TemporaryDirectory() as tmp:
            failures = compare(BASELINE, write_run(Path(tmp), self.manifest, records), 0.02, 0.8)
        self.assertTrue(any("overlap" in f for f in failures))

    def test_different_model_revision_is_refused(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["model_revisions"]["reranker_revision"] = "different"
        with tempfile.TemporaryDirectory() as tmp:
            failures = compare(BASELINE, write_run(Path(tmp), manifest, self.records), 0.02, 0.8)
        self.assertTrue(any("model_revisions" in f for f in failures))


if __name__ == "__main__":
    unittest.main()
