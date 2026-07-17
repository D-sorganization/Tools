# ruff: noqa: E501
"""Tests for rate limiting in the calculator API."""

import unittest

from flask import Flask
from web_applications.calculator.limiter import RateLimiter


class TestLimiter(unittest.TestCase):
    """Test suite for rate limiting functionality."""

    def setUp(self) -> None:
        """Set up the test application and limiter."""
        self.app = Flask(__name__)
        self.limiter = RateLimiter(limit=1, window=1)

        @self.app.route("/test")
        def test_route() -> str:
            if not self.limiter.is_allowed("127.0.0.1"):
                return "Too Many Requests", 429
            return "ok"

        self.client = self.app.test_client()

    def test_allow_within_limit(self) -> None:
        """Test that requests within the rate limit are allowed."""
        response = self.client.get("/test")
        self.assertEqual(response.status_code, 200)

    def test_deny_exceeding_limit(self) -> None:
        """Test that requests exceeding the rate limit are denied."""
        self.client.get("/test")
        response = self.client.get("/test")
        self.assertEqual(response.status_code, 429)

    def test_window_reset(self) -> None:
        """Test that the rate limit window resets after the specified duration."""
        import time

        self.client.get("/test")
        time.sleep(1.1)
        response = self.client.get("/test")
        self.assertEqual(response.status_code, 200)

    def test_independent_keys(self) -> None:
        """Test that limits are tracked independently for different keys."""
        self.assertTrue(self.limiter.is_allowed("1.1.1.1"))
        self.assertFalse(self.limiter.is_allowed("1.1.1.1"))
        self.assertTrue(self.limiter.is_allowed("2.2.2.2"))
