"""HTTP gateway behavior for the same-origin production companion."""

from __future__ import annotations

import json
import threading
from http.client import BadStatusLine, IncompleteRead
from types import MappingProxyType

from fastapi.testclient import TestClient

from rate_of_closure.application.regional_ground_authority_status import (
    AuthorityJobSnapshot,
    AuthorityJobStatus,
)
from rate_of_closure.application.regional_ground_authority_transport import (
    AuthorityHttpResponse,
)
from rate_of_closure.web_authority.capability import AuthorityCapability
from rate_of_closure.web_companion.app import create_companion_app
from rate_of_closure.web_companion.bundle import CompanionWebBundle
from rate_of_closure.web_distribution.asset_resolver import ResolvedWebAsset

_HOST = "127.0.0.1:52101"
_ORIGIN = f"http://{_HOST}"
_SECURE_POST_HEADERS = {
    "Origin": _ORIGIN,
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Dest": "empty",
    "Content-Type": "application/json",
}


class _RecordingTransport:
    def __init__(self) -> None:
        self.requests: list[tuple[str, str, bytes | None, int]] = []
        snapshot = AuthorityJobSnapshot(
            "bounded", "a" * 64, AuthorityJobStatus.QUEUED, 0, 1
        )
        self.response = AuthorityHttpResponse(
            202,
            "application/json",
            json.dumps(snapshot.to_wire(), separators=(",", ":")).encode(),
        )
        self.error: Exception | None = None
        self.closed = False

    def request(
        self, method: str, path: str, body: bytes | None, maximum_bytes: int
    ) -> AuthorityHttpResponse:
        self.requests.append((method, path, body, maximum_bytes))
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.closed = True


def _bundle() -> CompanionWebBundle:
    assets = MappingProxyType(
        {
            "assets/index-AbCd_123.js": ResolvedWebAsset(
                b"export {};", "text/javascript; charset=utf-8"
            )
        }
    )
    return CompanionWebBundle(
        "a" * 40,
        ResolvedWebAsset(b'<!doctype html><div id="root"></div>', "text/html"),
        assets,
    )


def _client(transport: _RecordingTransport) -> TestClient:
    return TestClient(
        create_companion_app(
            bundle=_bundle(), transport=transport, expected_host=_HOST
        ),
        base_url=_ORIGIN,
    )


def test_companion_serves_only_immutable_snapshot_with_security_headers() -> None:
    transport = _RecordingTransport()
    with _client(transport) as client:
        index = client.get("/")
        asset = client.get("/assets/index-AbCd_123.js")
        missing = client.get("/assets/missing.js")
    assert index.content.startswith(b"<!doctype html>")
    assert index.headers["cache-control"] == "no-store"
    assert asset.content == b"export {};"
    assert asset.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert asset.headers["x-content-type-options"] == "nosniff"
    assert asset.headers["referrer-policy"] == "no-referrer"
    assert asset.headers["x-frame-options"] == "DENY"
    assert "default-src 'none'" in asset.headers["content-security-policy"]
    assert not any(name.startswith("access-control-") for name in asset.headers)
    assert missing.status_code == 404
    assert transport.requests == []
    assert transport.closed is True


def test_companion_reconstructs_api_request_without_browser_credentials() -> None:
    transport = _RecordingTransport()
    with _client(transport) as client:
        response = client.post(
            "/api/rate-of-closure/v1/regional-ground/jobs",
            headers=_SECURE_POST_HEADERS,
            content=b'{"job":"bounded"}',
        )
    assert response.status_code == 202
    assert response.json()["job_id"] == "bounded"
    assert response.headers["cache-control"] == "no-store"
    assert transport.requests == [
        (
            "POST",
            "/api/rate-of-closure/v1/regional-ground/jobs",
            b'{"job":"bounded"}',
            4_096,
        )
    ]


def test_companion_rejects_before_transport_and_sanitizes_upstream_faults() -> None:
    transport = _RecordingTransport()
    with _client(transport) as client:
        rejected = client.post(
            "/api/rate-of-closure/v1/regional-ground/jobs",
            headers={"Content-Type": "application/json"},
            content=b"{}",
        )
        transport.response = AuthorityHttpResponse(
            302,
            "text/html",
            b"token=authority-secret port=54321",
        )
        failed = client.get("/api/rate-of-closure/v1/capabilities")
    assert rejected.status_code == 403
    assert transport.requests == [
        ("GET", "/api/rate-of-closure/v1/capabilities", None, 4_096)
    ]
    assert failed.status_code == 502
    assert b"authority-secret" not in failed.content
    assert b"54321" not in failed.content


def test_companion_enforces_content_contract_before_forwarding() -> None:
    transport = _RecordingTransport()
    with _client(transport) as client:
        wrong_media = client.post(
            "/api/rate-of-closure/v1/regional-ground/jobs",
            headers={**_SECURE_POST_HEADERS, "Content-Type": "text/plain"},
            content=b"{}",
        )
        encoded = client.post(
            "/api/rate-of-closure/v1/regional-ground/jobs",
            headers={**_SECURE_POST_HEADERS, "Content-Encoding": "gzip"},
            content=b"{}",
        )
        body_on_get = client.request(
            "GET",
            "/api/rate-of-closure/v1/capabilities",
            headers={"Content-Type": "application/json"},
            content=b"{}",
        )
    assert wrong_media.status_code == 415
    assert encoded.status_code == 415
    assert body_on_get.status_code == 400
    assert transport.requests == []


def test_companion_rejects_malformed_or_route_impossible_authority_json() -> None:
    transport = _RecordingTransport()
    with _client(transport) as client:
        transport.response = AuthorityHttpResponse(
            200, "application/json", b'{"available":true,"available":false}'
        )
        duplicate = client.get("/api/rate-of-closure/v1/capabilities")
        transport.response = AuthorityHttpResponse(
            202, "application/json", b'{"code":"job_not_found","detail":"no"}'
        )
        wrong_success = client.get("/api/rate-of-closure/v1/capabilities")
        transport.response = AuthorityHttpResponse(
            404, "application/json", b'{"code":"result_unavailable","detail":"no"}'
        )
        wrong_error = client.get(
            "/api/rate-of-closure/v1/regional-ground/jobs/bounded/result"
        )
    assert duplicate.status_code == 502
    assert wrong_success.status_code == 502
    assert wrong_error.status_code == 502


def test_companion_sanitizes_http_protocol_faults() -> None:
    transport = _RecordingTransport()
    with _client(transport) as client:
        for error in (IncompleteRead(b"authority-secret"), BadStatusLine("secret")):
            transport.error = error
            response = client.get("/api/rate-of-closure/v1/capabilities")
            assert response.status_code == 502
            assert response.json() == {"code": "authority_unavailable"}
            assert b"secret" not in response.content
            assert response.headers["cache-control"] == "no-store"


def test_companion_applies_security_envelope_to_uncommon_methods() -> None:
    transport = _RecordingTransport()
    with _client(transport) as client:
        for method in ("TRACE", "CONNECT", "BREW"):
            response = client.request(method, "/")
            assert response.status_code == 405
            assert response.json() == {"code": "not_found"}
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["x-frame-options"] == "DENY"
            assert "default-src 'none'" in response.headers["content-security-policy"]


def test_blocked_authority_request_does_not_block_static_gateway() -> None:
    transport = _RecordingTransport()
    entered = threading.Event()
    release = threading.Event()
    capability = AuthorityCapability.qualified()
    transport.response = AuthorityHttpResponse(
        200,
        "application/json",
        json.dumps(capability.to_wire(), separators=(",", ":")).encode(),
    )
    original_request = transport.request

    def blocked_request(method, path, body, maximum_bytes):
        entered.set()
        if not release.wait(5.0):
            raise TimeoutError("test release timeout")
        return original_request(method, path, body, maximum_bytes)

    transport.request = blocked_request  # type: ignore[method-assign]
    with _client(transport) as client:
        api_thread = threading.Thread(
            target=lambda: client.get("/api/rate-of-closure/v1/capabilities")
        )
        api_thread.start()
        assert entered.wait(2.0)
        static_result: list[int] = []
        static_thread = threading.Thread(
            target=lambda: static_result.append(client.get("/").status_code)
        )
        static_thread.start()
        static_thread.join(1.0)
        try:
            assert static_result == [200]
        finally:
            release.set()
            api_thread.join(3.0)
            static_thread.join(3.0)
