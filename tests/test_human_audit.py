import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("make_human_audit", Path(__file__).resolve().parents[1] / "scripts" / "make_human_audit.py")
make_human_audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_human_audit)


def write(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.original = (make_human_audit.AUDIT, make_human_audit.KEY)
        make_human_audit.AUDIT = self.directory
        make_human_audit.KEY = self.directory / "key.csv"
        labels = ["supported", "supported", "partially_supported", "contradicted"]
        write(make_human_audit.KEY, make_human_audit.KEY_FIELDS,
              [{"audit_id": str(i), "query_id": "q", "claim_id": f"C{i}", "judge_label": label, "judge_supporting": "[]"} for i, label in enumerate(labels, 1)])
        self.sample_fields = make_human_audit.SAMPLE_FIELDS

    def tearDown(self):
        make_human_audit.AUDIT, make_human_audit.KEY = self.original

    def filled(self, labels):
        path = self.directory / "filled.csv"
        write(path, self.sample_fields, [{**dict.fromkeys(self.sample_fields, ""), "audit_id": str(i), "reviewer_label": label} for i, label in enumerate(labels, 1)])
        return path

    def test_agreement_is_counted_against_the_key_and_unlabelled_rows_are_left_out(self):
        result = make_human_audit.score(self.filled(["supported", "partially_supported", "partially_supported", ""]))
        self.assertEqual(result["rows_in_sample"], 4)
        self.assertEqual(result["rows_labelled"], 3)
        self.assertAlmostEqual(result["agreement"], 2 / 3)
        self.assertEqual(result["agreement_by_judge_label"]["supported"], {"n": 2, "agreed": 1})

    def test_an_unknown_label_is_rejected(self):
        with self.assertRaises(ValueError):
            make_human_audit.score(self.filled(["fine", "", "", ""]))

    def test_the_blind_sample_has_no_judge_label_column(self):
        self.assertNotIn("judge_label", make_human_audit.SAMPLE_FIELDS)
        self.assertNotIn("judge_supporting", make_human_audit.SAMPLE_FIELDS)


if __name__ == "__main__":
    unittest.main()
