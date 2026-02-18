"""Security tests for calculator web application."""

import unittest

from web_applications.calculator.webapp import create_app


class TestSecurity(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.app.config.update({"TESTING": True})
        self.client = self.app.test_client()

    def test_input_too_large(self) -> None:
        # Create a very large expression (> 1000 characters)
        large_expression = "1+" * 1000 + "1"
        payload = {"operation": "evaluate", "expression": large_expression}
        response = self.client.post("/api/calculate", json=payload)

        if response.status_code == 200:
            self.fail("VULNERABILITY CONFIRMED: Large input accepted")

        self.assertEqual(
            response.status_code, 400, "Should reject excessively large input"
        )

    def test_security_headers(self) -> None:
        """Test that security headers are present in responses."""
        response = self.client.get("/")

        # HSTS
        self.assertIn("Strict-Transport-Security", response.headers)
        self.assertIn("max-age=31536000", response.headers["Strict-Transport-Security"])

        # CSP
        self.assertIn("Content-Security-Policy", response.headers)
        self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])

        # X-Content-Type-Options
        self.assertIn("X-Content-Type-Options", response.headers)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

        # X-Frame-Options
        self.assertIn("X-Frame-Options", response.headers)
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")


if __name__ == "__main__":
    unittest.main()
