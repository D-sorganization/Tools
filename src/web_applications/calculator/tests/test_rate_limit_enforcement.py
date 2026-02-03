"""Rate limit enforcement tests for calculator web application."""

import unittest

from web_applications.calculator.webapp import create_app


class TestRateLimitEnforcement(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        # Ensure TESTING is False (default)
        self.app.config.update({"TESTING": False})
        self.client = self.app.test_client()

    def test_rate_limit_exceeded(self) -> None:
        # Set low limit for testing
        # We access the limiter attached to the app
        self.app.limiter.limit = 2

        payload = {"operation": "evaluate", "expression": "1+1"}

        # 1st request - OK
        resp = self.client.post("/api/calculate", json=payload)
        self.assertEqual(resp.status_code, 200)

        # 2nd request - OK
        resp = self.client.post("/api/calculate", json=payload)
        self.assertEqual(resp.status_code, 200)

        # 3rd request - 429 Too Many Requests
        resp = self.client.post("/api/calculate", json=payload)
        self.assertEqual(resp.status_code, 429)
        self.assertIn("Rate limit exceeded", resp.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
