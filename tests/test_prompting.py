import unittest

from steam_review_rag.prompting import build_grounded_prompt


class PromptConstructionTests(unittest.TestCase):
    def test_prompt_numbers_and_normalizes_evidence(self):
        prompt = build_grounded_prompt(
            "  Which game is calm?  ",
            [
                {
                    "game_name": "Quiet Game",
                    "recommendation": "Recommended",
                    "review": "calm\n  and relaxing",
                },
                {
                    "game_name": "Busy Game",
                    "recommendation": "Not Recommended",
                    "review": "too loud",
                },
            ],
        )
        self.assertIn("[1] Game: Quiet Game", prompt)
        self.assertIn("[2] Game: Busy Game", prompt)
        self.assertIn("Review evidence: calm and relaxing", prompt)
        self.assertIn("Question: Which game is calm?", prompt)
        self.assertTrue(prompt.endswith("Answer only from the evidence above."))

    def test_prompt_truncates_review_evidence(self):
        prompt = build_grounded_prompt(
            "Question",
            [{"game_name": "Game", "recommendation": "Yes", "review": "abcdefgh"}],
            review_character_limit=4,
        )
        self.assertIn("Review evidence: abcd", prompt)
        self.assertNotIn("abcdefgh", prompt)


if __name__ == "__main__":
    unittest.main()
