import unittest
from unittest.mock import patch
import sys
import os

# Ensure Calculator package is in path (usually covered by running from root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Calculator.limiter import RateLimiter

class TestRateLimiter(unittest.TestCase):
    def test_allow_within_limit(self):
        limiter = RateLimiter(limit=5, window=60)
        for _ in range(5):
            self.assertTrue(limiter.is_allowed("1.2.3.4"))

    def test_deny_exceeding_limit(self):
        limiter = RateLimiter(limit=2, window=60)
        self.assertTrue(limiter.is_allowed("1.2.3.4"))
        self.assertTrue(limiter.is_allowed("1.2.3.4"))
        self.assertFalse(limiter.is_allowed("1.2.3.4"))

    def test_window_reset(self):
        limiter = RateLimiter(limit=1, window=60)

        with patch("time.time") as mock_time:
            # Window 1: 0-60. Time 10 falls in window 0.
            mock_time.return_value = 10
            self.assertTrue(limiter.is_allowed("1.2.3.4"))
            self.assertFalse(limiter.is_allowed("1.2.3.4"))

            # Window 2: 60-120. Time 70 falls in window 1.
            mock_time.return_value = 70
            self.assertTrue(limiter.is_allowed("1.2.3.4"))

    def test_independent_keys(self):
        limiter = RateLimiter(limit=1, window=60)
        self.assertTrue(limiter.is_allowed("A"))
        self.assertFalse(limiter.is_allowed("A"))
        self.assertTrue(limiter.is_allowed("B"))

if __name__ == '__main__':
    unittest.main()
