import unittest
from web_applications.calculator.webapp import create_app

class TestSecurityHeaders(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config.update({"TESTING": True})
        self.client = self.app.test_client()

    def test_security_headers_present(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        # CSP
        self.assertIn("Content-Security-Policy", response.headers)
        self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])

        # HSTS
        self.assertIn("Strict-Transport-Security", response.headers)
        self.assertIn("max-age=31536000", response.headers["Strict-Transport-Security"])

        # Permissions-Policy
        self.assertIn("Permissions-Policy", response.headers)
        expected_policy = "geolocation=(), camera=(), microphone=(), payment=(), usb=()"
        self.assertEqual(response.headers["Permissions-Policy"], expected_policy)

        # X-Content-Type-Options
        self.assertIn("X-Content-Type-Options", response.headers)
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

        # X-Frame-Options
        self.assertIn("X-Frame-Options", response.headers)
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")

if __name__ == "__main__":
    unittest.main()
