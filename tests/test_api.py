"""Contract tests for the REST service with a deterministic in-memory backend."""

import os
import unittest
from unittest import mock

import pandas as pd

try:
    from fastapi.testclient import TestClient

    from steam_review_rag.api import MAX_EVIDENCE, create_app
except ImportError as error:  # API extras are optional for the notebook environment.
    if os.environ.get("CI"):
        raise
    raise unittest.SkipTest(f"API dependencies not installed: {error}") from error


CORPUS = pd.DataFrame(
    {
        "raw_row_id": [101, 102, 103, 104, 105, 106],
        "game_name": ["Stardew Valley", "Stardew Valley", "Terraria", "Hades", "Hades", "Terraria"],
        "recommendation": ["Recommended", "Recommended", "Recommended", "Recommended", "Not Recommended", "Recommended"],
        "review": ["Relaxing farming after work", "Cozy and calm", "Digging and building",
                   "Fast roguelike combat", "Too repetitive for me", "Boss fights with friends"],
    }
)


class FakeBackend:
    """Ranks by word overlap so tests are deterministic and need no models."""

    device = "cpu"
    corpus = CORPUS

    def __init__(self):
        self.retrievals = 0
        self.generation_error: Exception | None = None

    def retrieve_and_rerank(self, question, evidence_count=None):
        self.retrievals += 1
        words = set(question.lower().split())
        frame = self.corpus.copy()
        frame["bi_score"] = frame["review"].str.lower().str.split().map(lambda w: len(words & set(w)) / 10)
        frame["rerank_score"] = frame["bi_score"] * 10
        frame["corpus_row_id"] = range(len(frame))
        return frame.sort_values(["rerank_score", "raw_row_id"], ascending=[False, True]).head(evidence_count or 5).reset_index(drop=True)

    def generate_answer(self, question, evidence, api_key, model_name):
        if self.generation_error:
            raise self.generation_error
        return "Players call it calm [1] and cozy [2]. Also [9]."


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.context = TestClient(create_app(lambda: self.backend))
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def test_root_redirects_to_the_interactive_docs(self):
        response = self.client.get("/", follow_redirects=False)
        self.assertIn(response.status_code, (302, 307))
        self.assertEqual(response.headers["location"], "/docs")

    def test_health_reports_corpus(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["corpus_reviews"], 6)
        self.assertEqual(body["games"], 3)

    def test_games_are_counted(self):
        games = {g["game_name"]: g["reviews"] for g in self.client.get("/games").json()}
        self.assertEqual(games, {"Stardew Valley": 2, "Terraria": 2, "Hades": 2})

    def test_search_returns_ranked_evidence_with_ids(self):
        body = self.client.post("/search", json={"question": "calm  and relaxing ", "evidence_count": 3}).json()
        self.assertEqual(body["question"], "calm and relaxing")
        self.assertEqual([e["rank"] for e in body["evidence"]], [1, 2, 3])
        self.assertEqual(body["evidence"][0]["review_id"], "102")  # shares "calm" and "and"
        self.assertFalse(body["cached"])

    def test_repeated_question_is_served_from_cache(self):
        payload = {"question": "Calm game", "evidence_count": 2}
        self.client.post("/search", json=payload)
        second = self.client.post("/search", json={"question": "calm   GAME", "evidence_count": 2}).json()
        self.assertTrue(second["cached"])
        self.assertEqual(self.backend.retrievals, 1)
        self.assertEqual(self.client.get("/health").json()["cache"]["hits"], 1)

    def test_answer_without_key_keeps_evidence(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GROQ_API_KEY", None)
            body = self.client.post("/answer", json={"question": "calm"}).json()
        self.assertEqual(body["generation_status"], "not_configured")
        self.assertIsNone(body["answer"])
        self.assertEqual(len(body["evidence"]), 5)

    def test_answer_flags_out_of_range_citations(self):
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            body = self.client.post("/answer", json={"question": "calm", "evidence_count": 3}).json()
        self.assertEqual(body["generation_status"], "generated")
        self.assertEqual(body["citation_check"], {
            "citation_count": 3, "invalid_citation_count": 1, "citation_numbering_valid": False,
        })

    def test_provider_failure_returns_safe_diagnostic(self):
        error = RuntimeError("secret upstream detail")
        self.backend.generation_error = error
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            body = self.client.post("/answer", json={"question": "calm"}).json()
        self.assertEqual(body["generation_status"], "failed")
        self.assertNotIn("secret", body["diagnostic"])
        self.assertTrue(body["evidence"])

    def test_invalid_requests_are_rejected(self):
        for payload in [{"question": "   "}, {"question": "x" * 601},
                        {"question": "ok", "evidence_count": MAX_EVIDENCE + 1},
                        {"question": "ok", "api_key": "client-supplied"}]:
            with self.subTest(payload=str(payload)[:40]):
                self.assertEqual(self.client.post("/search", json=payload).status_code, 422)

    def test_backend_failure_is_a_503(self):
        self.backend.retrieve_and_rerank = mock.Mock(side_effect=RuntimeError("index gone"))
        response = self.client.post("/search", json={"question": "calm"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("index gone", response.text)


if __name__ == "__main__":
    unittest.main()
