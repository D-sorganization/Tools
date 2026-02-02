"""Permissions policy tests for calculator web application."""
import unittest

from web_applications.calculator.webapp import create_app


class TestPermissionsPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        self.app.config.update({"TESTING": True})
        self.client = self.app.test_client()

    def test_permissions_policy_header_present(self) -> None:
        response = self.client.get("/")
        self.assertIn("Permissions-Policy", response.headers)
        expected_policy = "geolocation=(), camera=(), microphone=(), payment=(), usb=()"
        self.assertEqual(response.headers["Permissions-Policy"], expected_policy)


if __name__ == "__main__":
    unittest.main()
