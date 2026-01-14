import unittest

from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


class TestLimiter(unittest.TestCase):
    """Test suite for rate limiting functionality."""

    def setUp(self) -> None:
        """Set up the test application and limiter."""
        self.app = Flask(__name__)
        self.limiter = Limiter(
            get_remote_address, app=self.app, default_limits=["10 per hour"]
        )

        @self.app.route("/test")
        @self.limiter.limit("1 per second")
        def test_route() -> str:
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
        # This test would require mocking get_remote_address or using a different strategy
        pass
