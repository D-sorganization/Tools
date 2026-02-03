"""Rate limit proxy handling tests for calculator web application."""
import unittest

from web_applications.calculator.webapp import create_app


class TestRateLimitProxies(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app()
        # Ensure TESTING is False (default) so rate limiter is active
        self.app.config.update({"TESTING": False})
        self.client = self.app.test_client()

    def test_x_forwarded_for_different_clients(self) -> None:
        # Set limit to 1 per window
        self.app.limiter.limit = 1
        self.app.limiter.window = 60

        payload = {"operation": "evaluate", "expression": "1+1"}

        # 1st request from Client A (Direct via LB)
        # With ProxyFix(x_for=1), it trusts the last value.
        headers_a = {"X-Forwarded-For": "10.0.0.1"}
        resp_a = self.client.post("/api/calculate", json=payload, headers=headers_a)
        self.assertEqual(resp_a.status_code, 200, "Client A should be allowed")

        # 2nd request from Client B (Direct via LB)
        headers_b = {"X-Forwarded-For": "10.0.0.2"}
        resp_b = self.client.post("/api/calculate", json=payload, headers=headers_b)

        self.assertEqual(
            resp_b.status_code, 200, "Client B should be allowed (distinct IP)"
        )

    def test_x_forwarded_for_proxy_handling(self) -> None:
        # Test that ProxyFix is active and processing the header.
        # With x_for=1, it uses the last IP in the list.
        self.app.limiter.limit = 1

        payload = {"operation": "evaluate", "expression": "1+1"}

        # Client C coming through Proxy1
        headers_c = {"X-Forwarded-For": "10.0.0.3, 192.168.1.1"}
        resp_c = self.client.post("/api/calculate", json=payload, headers=headers_c)
        self.assertEqual(resp_c.status_code, 200)

        # Client C again (same path) -> Blocked
        resp_c2 = self.client.post("/api/calculate", json=payload, headers=headers_c)
        self.assertEqual(resp_c2.status_code, 429)

        # Client D coming through DIFFERENT Proxy2
        # This confirms ProxyFix is extracting the last IP (192.168.1.2)
        # and treating it as the client/source.
        headers_d = {"X-Forwarded-For": "10.0.0.4, 192.168.1.2"}
        resp_d = self.client.post("/api/calculate", json=payload, headers=headers_d)
        self.assertEqual(
            resp_d.status_code, 200, "Client D (different proxy) should be distinct"
        )


if __name__ == "__main__":
    unittest.main()
