"""Claim splitting, verdict validation and aggregation counterexamples, no network."""

import csv
import importlib.util
import inspect
import json
from pathlib import Path
import random
import tempfile
import unittest

from steam_review_rag import answer_eval as ae
from steam_review_rag.prompting import build_grounded_prompt


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluate_answers", ROOT / "scripts" / "evaluate_answers.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def verdict_json(labels, abstains=False, supporting=None):
    supporting = supporting or {}
    return json.dumps({
        "claims": [{"id": f"C{i}", "label": label, "supporting": supporting.get(i, [])}
                   for i, label in enumerate(labels, start=1)],
        "abstains": abstains,
    })


class SplitClaimsTests(unittest.TestCase):
    def test_trailing_citation_stays_with_its_sentence(self):
        claims = ae.split_claims("Players enjoy the puzzles. [1] The story is short. [2][3]")
        self.assertEqual([c["citations"] for c in claims], [[1], [2, 3]])
        self.assertEqual(len(claims), 2)

    def test_bullets_markdown_and_table_separators_are_removed(self):
        answer = "**Summary**\n- Great co-op [1]\n- Bugs on launch [2]\n| a | b |\n|---|---|"
        claims = ae.split_claims(answer)
        self.assertEqual([c["text"] for c in claims][:2], ["Great co-op [1]", "Bugs on launch [2]"])
        self.assertNotIn("---", " ".join(c["text"] for c in claims))

    def test_fullwidth_bracket_citations_are_recognised(self):
        claims = ae.split_claims("Players enjoy the soundtrack and the art style \u30102\u3011.")
        self.assertEqual(claims[0]["citations"], [2])

    def test_a_title_fragment_is_not_a_claim(self):
        claims = ae.split_claims("**Critics**\nSome players found the story short and the combat repetitive. [2]")
        self.assertEqual([c["citations"] for c in claims], [[2]])

    def test_a_short_complete_sentence_is_still_a_claim(self):
        self.assertEqual(len(ae.split_claims("It is great.")), 1)

    def test_citation_only_text_is_not_a_claim(self):
        self.assertEqual(ae.split_claims("[1][2]"), [])

    def test_review_truncation_matches_the_generation_prompt(self):
        default = inspect.signature(build_grounded_prompt).parameters["review_character_limit"].default
        self.assertEqual(ae.REVIEW_CHARACTER_LIMIT, default)
        row = {"game_name": "G", "recommendation": "Recommended", "review": "x " * 2000}
        prompt = build_grounded_prompt("q", [row])
        self.assertIn(ae.evidence_block(1, row), prompt)


class ParseTests(unittest.TestCase):
    def test_accepts_fenced_reply(self):
        text = "```json\n" + verdict_json(["supported"], supporting={1: [2]}) + "\n```"
        parsed = ae.parse_judge_response(text, ["C1"], 3)
        self.assertEqual(parsed["claims"]["C1"]["supporting"], [2])

    def test_rejects_bad_label_range_missing_and_duplicate_claims(self):
        for text, ids in [
            (verdict_json(["great"]), ["C1"]),
            (verdict_json(["supported"], supporting={1: [9]}), ["C1"]),
            (verdict_json(["supported"]), ["C1", "C2"]),
            (verdict_json(["supported", "supported"]).replace("C2", "C1"), ["C1"]),
            ("no json here", ["C1"]),
        ]:
            with self.assertRaises(ae.JudgeParseError):
                ae.parse_judge_response(text, ids, 3)

    def test_judge_retries_once_then_gives_up(self):
        calls = []

        def broken(system, user):
            calls.append(1)
            return "oops"

        claims = ae.split_claims("A claim. [1]")
        failures = []
        self.assertIsNone(ae.judge_answer("q", ["[1] e"], claims, broken, failures=failures))
        self.assertEqual(len(calls), 3)
        self.assertEqual([f["attempt"] for f in failures], [1, 2, 3])
        self.assertEqual(failures[0]["tail"], "oops")

    def test_only_the_retries_carry_the_reminder_note(self):
        prompts = []

        def reply(system, user):
            prompts.append(user)
            return "not json" if len(prompts) == 1 else verdict_json(["supported"], supporting={1: [1]})

        claims = ae.split_claims("A claim. [1]")
        self.assertIsNotNone(ae.judge_answer("q", ["[1] e"], claims, reply))
        self.assertNotIn(ae.RETRY_NOTE, prompts[0])
        self.assertTrue(prompts[1].endswith(ae.RETRY_NOTE))


class MetricTests(unittest.TestCase):
    def record(self, category, labels, citations, abstains=False, supporting=None):
        claims = [{"claim_id": f"C{i}", "text": "t", "citations": c}
                  for i, c in enumerate(citations, start=1)]
        verdict = ae.parse_judge_response(
            verdict_json(labels, abstains, supporting), [c["claim_id"] for c in claims], 5)
        return {"query_id": "q", "category": category, "verdict": verdict,
                "claims": ae.judged_claims(claims, verdict)}

    def test_valid_citation_number_does_not_make_an_unsupported_claim_correct(self):
        record = self.record("game_topic", ["unsupported"], [[1]], supporting={1: [1]})
        self.assertFalse(record["claims"][0]["citation_ok"])

    def test_citation_must_overlap_judged_support(self):
        right = self.record("game_topic", ["supported"], [[2]], supporting={1: [2]})
        wrong = self.record("game_topic", ["supported"], [[1]], supporting={1: [2]})
        self.assertTrue(right["claims"][0]["citation_ok"])
        self.assertFalse(wrong["claims"][0]["citation_ok"])

    def test_uncited_supported_claim_is_excluded_from_citation_rate(self):
        record = self.record("game_topic", ["supported", "supported"], [[], [1]], supporting={2: [1]})
        row = ae.summarize([record])[0]
        self.assertEqual(row["citation_correct_rate"], 1.0)
        self.assertEqual(row["cited_claim_share"], 0.5)

    def test_judge_failures_are_counted_not_dropped(self):
        failed = {"query_id": "q", "category": "unsupported", "verdict": None, "claims": []}
        ok = self.record("unsupported", ["unsupported"], [[]], abstains=False)
        row = [r for r in ae.summarize([failed, ok]) if r["category"] == "unsupported"][0]
        self.assertEqual((row["answers"], row["answers_judged"], row["judge_failures"]), (2, 1, 1))
        self.assertEqual(row["abstain_rate"], 0.0)
        self.assertEqual(row["answers_with_unsupported_claim"], 1)

    def test_bootstrap_resamples_answers_and_is_reproducible(self):
        pairs = [(3, 4), (0, 2), (5, 5), (1, 3)]
        first = ae.bootstrap_ratio(pairs)
        self.assertEqual(first, ae.bootstrap_ratio(pairs))
        self.assertAlmostEqual(first[0], 9 / 14)
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])
        self.assertEqual(ae.bootstrap_ratio([(0, 0)]), (None, None, None))

    def test_wilson_interval_stays_inside_zero_and_one(self):
        point, low, high = ae.wilson(10, 10)
        self.assertEqual(point, 1.0)
        self.assertTrue(0 <= low < 1 and high <= 1)
        self.assertEqual(ae.wilson(0, 0), (None, None, None))


class RefreshTests(unittest.TestCase):
    def test_stored_record_gets_new_citations_and_loses_its_heading(self):
        claims = [{"claim_id": "C1", "text": "Critics", "citations": []},
                  {"claim_id": "C2", "text": "The soundtrack is praised \u30102\u3011.", "citations": []}]
        verdict = ae.parse_judge_response(
            verdict_json(["unsupported", "supported"], supporting={2: [2]}), ["C1", "C2"], 5)
        stored = {"query_id": "q", "category": "game_topic", "verdict": verdict,
                  "claims": [{**c, **verdict["claims"][c["claim_id"]], "cited": False, "citation_ok": False}
                             for c in claims]}
        record = ae.refresh_record(stored)
        self.assertEqual([c["claim_id"] for c in record["claims"]], ["C2"])
        self.assertEqual(record["claims"][0]["citations"], [2])
        self.assertTrue(record["claims"][0]["citation_ok"])
        self.assertNotIn("C1", record["verdict"]["claims"])
        self.assertEqual(ae.refresh_record({"verdict": None, "claims": []})["claims"], [])


class AgreementTests(unittest.TestCase):
    def test_kappa_corrects_for_a_majority_class(self):
        first = ["supported"] * 9 + ["unsupported"]
        second = ["supported"] * 10
        result = ae.label_agreement(first, second)
        self.assertEqual(result["agreement"], 0.9)
        self.assertEqual(result["kappa"], 0.0)

    def test_identical_single_label_has_undefined_kappa(self):
        self.assertIsNone(ae.label_agreement(["supported"] * 3, ["supported"] * 3)["kappa"])

    def test_length_mismatch_is_an_error(self):
        with self.assertRaises(ValueError):
            ae.label_agreement(["supported"], [])

    def test_audit_sample_keeps_rare_labels(self):
        rows = [{"label": "supported", "i": i} for i in range(50)] + [{"label": "contradicted", "i": 99}]
        sample = ae.audit_sample(rows, 10)
        self.assertEqual(len(sample), 10)
        self.assertIn("contradicted", {r["label"] for r in sample})


class RunnerTests(unittest.TestCase):
    def item(self, review_text="Great puzzles."):
        return {"query_id": "q1", "question": "Is it good?", "answer": "It has great puzzles. [1] Nothing else is said.",
                "top_5": {"reranked": [{"review_id": "7", "game_name": "G",
                                         "review_sha256": runner.review_hash(review_text)}]}}

    def test_evidence_hash_mismatch_stops_the_run(self):
        by_id = type("ById", (), {"loc": {"7": {"review": "Changed text"}}})()
        with self.assertRaises(ValueError):
            runner.evidence_for(self.item(), by_id)

    def test_judge_item_runs_two_passes_and_records_failures(self):
        calls = []

        def complete(system, user):
            calls.append(user)
            return verdict_json(["supported", "unsupported"], supporting={1: [1]})

        rows = [{"review": "Great puzzles.", "game_name": "G", "recommendation": "Recommended"}]
        outcome = runner.judge_item(self.item(), rows, complete, random.Random(1))
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(outcome["verdict"])
        self.assertIsNotNone(outcome["second"])

        def broken(system, user):
            raise RuntimeError("provider down")

        failed = runner.judge_item(self.item(), rows, broken, random.Random(1))
        self.assertIsNone(failed["verdict"])
        self.assertEqual(failed["error"], "RuntimeError")

    def test_backoff_waits_the_advised_time_then_succeeds(self):
        class Limited(Exception):
            status_code = 429
            response = type("R", (), {"headers": {"retry-after": "7"}})()

        sleeps, outcomes = [], [Limited(), "ok"]

        def call():
            result = outcomes.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        self.assertEqual(runner.call_with_backoff(call, sleep=sleeps.append), "ok")
        self.assertEqual(sleeps, [8])

    def test_backoff_gives_up_on_a_daily_limit_and_ignores_other_errors(self):
        class Daily(Exception):
            status_code = 429
            response = type("R", (), {"headers": {"retry-after": "7200"}})()

        def daily():
            raise Daily()

        with self.assertRaises(runner.RateLimitExhausted):
            runner.call_with_backoff(daily, sleep=lambda s: None)
        calls = []

        def broken():
            calls.append(1)
            raise ValueError("bad request")

        with self.assertRaises(ValueError):
            runner.call_with_backoff(broken, sleep=lambda s: None)
        self.assertEqual(len(calls), 1)

    def test_pending_work_keeps_finished_verdicts(self):
        per_query = [{"query_id": "a", "generation_status": "generated"},
                     {"query_id": "b", "generation_status": "generated"},
                     {"query_id": "c", "generation_status": "failed"}]
        prior = {"a": {"verdict": {"claims": {}, "abstains": False}}, "b": {"verdict": None}}
        self.assertEqual(runner.pending_work(per_query, prior, True), (["c"], ["b", "c"]))
        self.assertEqual(runner.pending_work(per_query, prior, False), ([], ["b"]))

    def test_stability_uses_only_answers_judged_twice(self):
        first = ae.parse_judge_response(verdict_json(["supported"]), ["C1"], 5)
        second = ae.parse_judge_response(verdict_json(["unsupported"], abstains=True), ["C1"], 5)
        twice = {"verdict": first, "second": second}
        once = {"verdict": first, "second": None}
        claims, abstains = runner.stability([twice, once, {"verdict": None, "second": None}])
        self.assertEqual((claims["n"], claims["agreement"], abstains["n"]), (1, 0.0, 1))

    def test_summary_flags_incomplete_coverage_and_the_unreliable_abstain_flag(self):
        record = {"query_id": "q", "category": "unsupported",
                  "claims": [], "verdict": {"claims": {}, "abstains": False}}
        rows = ae.summarize([record, {**record, "verdict": None}])
        info = {"date": "2026-10-30", "generator": "g", "judge": "j", "questions": 2, "unanswered": 0,
                "unjudged": 1, "stability": "s", "audit": "a"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.md"
            runner.write_summary(path, rows, info)
            text = path.read_text()
        self.assertIn("**Incomplete.**", text)
        self.assertIn("Judge abstain flag", text)
        self.assertIn("wrong when checked by hand", text)

    def test_audit_scoring_ignores_unlabelled_rows_and_rejects_unknown_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim_audit_sample.csv"
            rows = [{k: "" for k in runner.AUDIT_FIELDS} for _ in range(3)]
            rows[0].update(judge_label="supported", reviewer_label="supported")
            rows[1].update(judge_label="supported", reviewer_label="unsupported")
            rows[2].update(judge_label="unsupported")
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=runner.AUDIT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            result = runner.score_audit(path)
            self.assertEqual((result["rows_labelled"], result["rows_in_sample"], result["agreement"]), (2, 3, 0.5))
            rows[0]["reviewer_label"] = "fine"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=runner.AUDIT_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaises(ValueError):
                runner.score_audit(path)


if __name__ == "__main__":
    unittest.main()
