"""Tests for web_applications/calculator/limiter.py — RateLimiter.

Covers the thread-safe fixed-window rate limiting logic.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path

import pytest

# Import limiter directly from file to skip web_applications/__init__.py (requires flask)
_limiter_path = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "web_applications"
    / "calculator"
    / "limiter.py"
)
_spec = importlib.util.spec_from_file_location("limiter", _limiter_path)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
RateLimiter = _mod.RateLimiter


class TestRateLimiterBasic:
    """Basic correctness tests for RateLimiter."""

    def test_allows_first_request(self) -> None:
        rl = RateLimiter(limit=5, window=60)
        assert rl.is_allowed("user1") is True

    def test_allows_up_to_limit(self) -> None:
        rl = RateLimiter(limit=3, window=60)
        assert rl.is_allowed("u") is True
        assert rl.is_allowed("u") is True
        assert rl.is_allowed("u") is True

    def test_blocks_over_limit(self) -> None:
        rl = RateLimiter(limit=2, window=60)
        rl.is_allowed("u")
        rl.is_allowed("u")
        assert rl.is_allowed("u") is False

    def test_different_keys_are_independent(self) -> None:
        rl = RateLimiter(limit=1, window=60)
        assert rl.is_allowed("alice") is True
        assert rl.is_allowed("bob") is True
        # alice is now blocked, bob still gets one more
        assert rl.is_allowed("alice") is False
        assert rl.is_allowed("bob") is False

    def test_limit_of_one_blocks_second_request(self) -> None:
        rl = RateLimiter(limit=1, window=60)
        assert rl.is_allowed("ip") is True
        assert rl.is_allowed("ip") is False

    def test_large_limit_allows_many(self) -> None:
        rl = RateLimiter(limit=1000, window=60)
        for _ in range(100):
            assert rl.is_allowed("power-user") is True

    def test_initialization_state(self) -> None:
        rl = RateLimiter(limit=10, window=30)
        assert rl.limit == 10
        assert rl.window == 30
        assert rl.hits == {}

    def test_require_key_not_none(self) -> None:
        rl = RateLimiter(limit=5, window=60)
        with pytest.raises(AssertionError):
            rl.is_allowed(None)  # type: ignore[arg-type]


class TestRateLimiterWindowReset:
    """Tests for window expiry and reset behavior."""

    def test_new_window_resets_count(self) -> None:
        """After window expires, limit counter should reset."""
        rl = RateLimiter(limit=2, window=1)
        assert rl.is_allowed("u") is True
        assert rl.is_allowed("u") is True
        assert rl.is_allowed("u") is False  # blocked

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again in new window
        assert rl.is_allowed("u") is True

    def test_global_window_clears_all_keys(self) -> None:
        """When global window rolls, all keys should be cleared."""
        rl = RateLimiter(limit=1, window=1)
        rl.is_allowed("alice")
        rl.is_allowed("bob")
        assert rl.is_allowed("alice") is False
        assert rl.is_allowed("bob") is False

        time.sleep(1.1)

        # Both should be reset
        assert rl.is_allowed("alice") is True
        assert rl.is_allowed("bob") is True


class TestRateLimiterThreadSafety:
    """Thread safety tests for RateLimiter."""

    def test_concurrent_requests_do_not_exceed_limit(self) -> None:
        """Under heavy concurrency, total allowed requests must not exceed limit."""
        limit = 50
        rl = RateLimiter(limit=limit, window=60)
        results: list[bool] = []
        lock = threading.Lock()

        def make_request() -> None:
            result = rl.is_allowed("shared_key")
            with lock:
                results.append(result)

        threads = [threading.Thread(target=make_request) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        allowed_count = sum(1 for r in results if r)
        assert allowed_count == limit, (
            f"Expected exactly {limit} allowed requests, got {allowed_count}"
        )

    def test_concurrent_different_keys(self) -> None:
        """Different keys under concurrency should each get their own limit."""
        rl = RateLimiter(limit=5, window=60)
        results: dict[str, list[bool]] = {"a": [], "b": []}
        lock = threading.Lock()

        def make_requests(key: str) -> None:
            for _ in range(10):
                r = rl.is_allowed(key)
                with lock:
                    results[key].append(r)

        threads = [
            threading.Thread(target=make_requests, args=("a",)),
            threading.Thread(target=make_requests, args=("b",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sum(results["a"]) == 5
        assert sum(results["b"]) == 5
