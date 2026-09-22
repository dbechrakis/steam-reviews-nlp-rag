import unittest

from steam_review_rag.diagnostics import generation_diagnostic


class RateLimitError(Exception):
    status_code = 429
    body = {"error": {"code": "rate_limit_exceeded"}}


class GenerationDiagnosticTests(unittest.TestCase):
    def test_rate_limit_message_keeps_evidence_available(self):
        message = generation_diagnostic(RateLimitError("secret response"), "openai/gpt-oss-120b")
        self.assertIn("HTTP 429", message)
        self.assertIn("retrieved reviews remain available", message)
        self.assertNotIn("secret response", message)

    def test_unexpected_error_does_not_expose_exception_text(self):
        message = generation_diagnostic(RuntimeError("api-key-should-not-leak"), "unknown-model")
        self.assertNotIn("api-key-should-not-leak", message)
        self.assertIn("could not be generated", message)


if __name__ == "__main__":
    unittest.main()
