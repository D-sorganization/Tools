import os
import sys
import unittest

# Add the repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from calculator.webapp import create_app


class TestSecurityHeaders(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.app.config.update({"TESTING": True})
        self.client = self.app.test_client()

    def test_security_headers_present(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        # Check for X-Content-Type-Options
        self.assertEqual(
            response.headers.get("X-Content-Type-Options"),
            "nosniff",
            "Missing X-Content-Type-Options header",
        )

        # Check for X-Frame-Options
        self.assertEqual(
            response.headers.get("X-Frame-Options"),
            "SAMEORIGIN",
            "Missing X-Frame-Options header",
        )

        # Check for Referrer-Policy
        self.assertEqual(
            response.headers.get("Referrer-Policy"),
            "strict-origin-when-cross-origin",
            "Missing Referrer-Policy header",
        )

        # Check for Content-Security-Policy
        csp = response.headers.get("Content-Security-Policy")
        self.assertIsNotNone(csp, "Missing Content-Security-Policy header")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("object-src 'none'", csp)


if __name__ == "__main__":
    unittest.main()
