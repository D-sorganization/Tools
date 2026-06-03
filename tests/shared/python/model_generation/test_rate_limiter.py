from __future__ import annotations

import logging
import urllib.error
import urllib.request
from typing import Any

import model_generation.library._rate_limiter as rate_limiter
import pytest


class _Response:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


def _http_error(
    code: int,
    headers: dict[str, str] | None = None,
) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.github.test/search",
        code,
        "test error",
        headers or {},
        None,
    )


def test_extract_rate_limit_info_parses_valid_headers() -> None:
    response = _Response(
        {
            "X-RateLimit-Remaining": "42",
            "X-RateLimit-Limit": "60",
            "X-RateLimit-Reset": "1710000000",
        }
    )

    assert rate_limiter.extract_rate_limit_info(response) == {
        "remaining": 42,
        "limit": 60,
        "reset_epoch": 1710000000,
    }


def test_extract_rate_limit_info_ignores_missing_and_invalid_headers() -> None:
    response = _Response(
        {
            "X-RateLimit-Remaining": "not-an-int",
            "X-RateLimit-Limit": "",
            "X-RateLimit-Reset": "1710000000",
        }
    )

    assert rate_limiter.extract_rate_limit_info(response) == {
        "remaining": None,
        "limit": None,
        "reset_epoch": 1710000000,
    }
    assert rate_limiter.extract_rate_limit_info(object()) == {
        "remaining": None,
        "limit": None,
        "reset_epoch": None,
    }


def test_log_rate_limit_status_records_remaining_and_reset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=rate_limiter.logger.name)

    rate_limiter.log_rate_limit_status(
        "https://api.github.test/search/repositories",
        {"remaining": 3, "limit": 60, "reset_epoch": 1710000000},
        200,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any("3/60 remaining" in message for message in messages)
    assert any("Rate-limit resets at" in message for message in messages)


def test_log_rate_limit_status_warns_for_429_without_headers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=rate_limiter.logger.name)

    rate_limiter.log_rate_limit_status(
        "https://api.github.test/search/repositories",
        {"remaining": None, "limit": None, "reset_epoch": None},
        429,
    )

    assert "Rate-limit exceeded (429)" in caplog.text


def test_make_request_with_backoff_adds_headers_and_logs_success(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen_requests: list[tuple[str, dict[str, str], float]] = []
    response = _Response(
        {
            "X-RateLimit-Remaining": "59",
            "X-RateLimit-Limit": "60",
        }
    )

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        seen_requests.append(
            (
                request.full_url,
                dict(request.header_items()),
                timeout,
            )
        )
        return response

    monkeypatch.setattr(rate_limiter.urllib.request, "urlopen", fake_urlopen)
    caplog.set_level(logging.INFO, logger=rate_limiter.logger.name)

    result = rate_limiter.make_request_with_backoff(
        "https://api.github.test/search",
        headers={"Authorization": "Bearer token"},
    )

    assert result is response
    assert seen_requests == [
        (
            "https://api.github.test/search",
            {"Authorization": "Bearer token"},
            10,
        )
    ]
    assert "59/60 remaining" in caplog.text


def test_make_request_with_backoff_retries_429_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []
    response = _Response({"X-RateLimit-Remaining": "1", "X-RateLimit-Limit": "60"})

    def fake_urlopen(
        _request: urllib.request.Request,
        timeout: float,
    ) -> _Response:
        nonlocal attempts
        assert timeout == 10
        attempts += 1
        if attempts == 1:
            raise _http_error(429, {"X-RateLimit-Remaining": "0"})
        return response

    monkeypatch.setattr(rate_limiter.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(rate_limiter.time, "sleep", sleeps.append)

    result = rate_limiter.make_request_with_backoff(
        "https://api.github.test/search",
        max_retries=2,
        initial_backoff=0.5,
        max_backoff=10,
    )

    assert result is response
    assert attempts == 2
    assert sleeps == [0.5]


def test_make_request_with_backoff_raises_rate_limit_error_after_final_429(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sleeps: list[float] = []

    def fake_urlopen(
        _request: urllib.request.Request,
        timeout: float,
    ) -> _Response:
        assert timeout == 10
        raise _http_error(
            429,
            {
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Limit": "60",
                "X-RateLimit-Reset": "1710000000",
            },
        )

    monkeypatch.setattr(rate_limiter.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(rate_limiter.time, "sleep", sleeps.append)
    caplog.set_level(logging.WARNING, logger=rate_limiter.logger.name)

    with pytest.raises(
        rate_limiter.RateLimitError,
        match="Rate limit exceeded after 3 attempts",
    ):
        rate_limiter.make_request_with_backoff(
            "https://api.github.test/search",
            max_retries=3,
            initial_backoff=1.0,
            max_backoff=1.5,
        )

    assert sleeps == [1.0, 1.5]
    assert "Rate limited (attempt 1/3)" in caplog.text


def test_make_request_with_backoff_does_not_retry_non_429_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    expected = _http_error(404)

    def fake_urlopen(
        _request: urllib.request.Request,
        timeout: float,
    ) -> _Response:
        assert timeout == 10
        nonlocal calls
        calls += 1
        raise expected

    monkeypatch.setattr(rate_limiter.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        rate_limiter.make_request_with_backoff(
            "https://api.github.test/missing",
            max_retries=3,
        )

    assert exc_info.value is expected
    assert calls == 1


def test_make_request_with_backoff_retries_network_error_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []
    response = _Response()

    def fake_urlopen(
        _request: urllib.request.Request,
        timeout: float,
    ) -> _Response:
        assert timeout == 10
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.URLError("temporary DNS failure")
        return response

    monkeypatch.setattr(rate_limiter.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(rate_limiter.time, "sleep", sleeps.append)

    assert (
        rate_limiter.make_request_with_backoff(
            "https://api.github.test/search",
            max_retries=2,
            initial_backoff=0.25,
        )
        is response
    )
    assert attempts == 2
    assert sleeps == [0.25]


def test_make_request_with_backoff_reraises_final_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = OSError("connection reset")

    def fake_urlopen(
        _request: urllib.request.Request,
        timeout: float,
    ) -> Any:
        assert timeout == 10
        raise expected

    monkeypatch.setattr(rate_limiter.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(OSError) as exc_info:
        rate_limiter.make_request_with_backoff(
            "https://api.github.test/search",
            max_retries=1,
        )

    assert exc_info.value is expected


def test_make_request_with_backoff_zero_retries_raises_rate_limit_error() -> None:
    with pytest.raises(
        rate_limiter.RateLimitError,
        match="Request failed: no retries remaining",
    ):
        rate_limiter.make_request_with_backoff(
            "https://api.github.test/search",
            max_retries=0,
        )
