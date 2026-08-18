"""Fail-closed request contracts for the source production companion."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from rate_of_closure.web_companion.contracts import (
    CompanionRequest,
    CompanionRequestRejected,
    CompanionRouteKind,
    classify_companion_request,
)

_HOST = "127.0.0.1:52101"
_ORIGIN = f"http://{_HOST}"


def _headers(**overrides: str) -> Mapping[str, str]:
    return {
        "host": _HOST,
        "accept": "application/json",
        **overrides,
    }


def _request(
    method: str,
    path: bytes,
    *,
    headers: Mapping[str, str] | None = None,
    query: bytes = b"",
) -> CompanionRequest:
    return CompanionRequest(
        method=method,
        raw_path=path,
        query_string=query,
        headers=_headers() if headers is None else headers,
    )


@pytest.mark.parametrize("path", [b"/", b"/index.html"])
def test_companion_routes_only_canonical_index_paths(path: bytes) -> None:
    route = classify_companion_request(_request("GET", path), expected_host=_HOST)
    assert route.kind is CompanionRouteKind.INDEX
    assert route.upstream_path is None


def test_companion_routes_one_declared_asset_without_decoding() -> None:
    route = classify_companion_request(
        _request("GET", b"/assets/index-AbCd_123.js"), expected_host=_HOST
    )
    assert route.kind is CompanionRouteKind.ASSET
    assert route.asset_path == "assets/index-AbCd_123.js"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", b"/api/rate-of-closure/v1/capabilities"),
        ("POST", b"/api/rate-of-closure/v1/regional-ground/jobs"),
        ("POST", b"/api/rate-of-closure/v1/regional-ground/job-preparations"),
        ("GET", b"/api/rate-of-closure/v1/regional-ground/jobs/job-1"),
        ("POST", b"/api/rate-of-closure/v1/regional-ground/jobs/job-1/cancel"),
        ("GET", b"/api/rate-of-closure/v1/regional-ground/jobs/job-1/result"),
    ],
)
def test_companion_allows_only_fixed_authority_routes(method: str, path: bytes) -> None:
    headers = _headers()
    if method == "POST":
        headers = _headers(
            origin=_ORIGIN,
            **{
                "sec-fetch-site": "same-origin",
                "sec-fetch-mode": "cors",
                "sec-fetch-dest": "empty",
            },
        )
    route = classify_companion_request(
        _request(method, path, headers=headers), expected_host=_HOST
    )
    assert route.kind is CompanionRouteKind.API
    assert route.upstream_path == path.decode("ascii")


@pytest.mark.parametrize(
    "path",
    [
        b"/%2e%2e/secret",
        b"/assets/../secret",
        b"/assets//index.js",
        b"/assets\\index.js",
        b"/rate-of-closure-assets.v1.json",
        b"/api/rate-of-closure/v1/unknown",
        b"/api/rate-of-closure/v1/regional-ground/jobs/bad%3Aid",
    ],
)
def test_companion_rejects_unknown_or_ambiguous_paths(path: bytes) -> None:
    with pytest.raises(CompanionRequestRejected):
        classify_companion_request(_request("GET", path), expected_host=_HOST)


@pytest.mark.parametrize(
    "headers",
    [
        _headers(host="localhost:52101"),
        _headers(authorization="Bearer browser-secret"),
        _headers(cookie="session=browser-secret"),
        _headers(forwarded="for=127.0.0.1"),
        _headers(**{"x-forwarded-host": _HOST}),
        _headers(**{"access-control-request-method": "POST"}),
    ],
)
def test_companion_rejects_browser_credentials_and_forwarding(
    headers: Mapping[str, str],
) -> None:
    with pytest.raises(CompanionRequestRejected):
        classify_companion_request(
            _request("GET", b"/", headers=headers), expected_host=_HOST
        )


@pytest.mark.parametrize(
    "headers",
    [
        _headers(),
        _headers(origin="http://evil.invalid"),
        _headers(origin=_ORIGIN, **{"sec-fetch-site": "cross-site"}),
        _headers(
            origin=_ORIGIN,
            **{"sec-fetch-site": "same-origin", "sec-fetch-mode": "navigate"},
        ),
    ],
)
def test_state_changes_require_exact_same_origin_fetch_metadata(
    headers: Mapping[str, str],
) -> None:
    with pytest.raises(CompanionRequestRejected):
        classify_companion_request(
            _request(
                "POST",
                b"/api/rate-of-closure/v1/regional-ground/jobs",
                headers=headers,
            ),
            expected_host=_HOST,
        )


def test_companion_rejects_queries_and_preflight() -> None:
    with pytest.raises(CompanionRequestRejected):
        classify_companion_request(
            _request("GET", b"/", query=b"cache=bust"), expected_host=_HOST
        )
    with pytest.raises(CompanionRequestRejected):
        classify_companion_request(
            _request("OPTIONS", b"/api/rate-of-closure/v1/capabilities"),
            expected_host=_HOST,
        )
