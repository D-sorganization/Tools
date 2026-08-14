"""Wire tests for the bounded authenticated loopback authority transport."""

from __future__ import annotations

import pytest

from rate_of_closure.application import regional_ground_authority_transport
from rate_of_closure.application.regional_ground_authority_transport import (
    AuthorityHttpResponse,
    LoopbackAuthorityHttpTransport,
)
from rate_of_closure.web_authority.api import JOB_COLLECTION_PATH
from rate_of_closure.web_authority.runtime import LOOPBACK_HOST, AuthorityRuntime

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_http_transport_uses_fixed_loopback_and_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "runtime-secret"
    captured: dict[str, object] = {}

    class _Response:
        status = 200

        @staticmethod
        def getheader(name: str, default: str | None = None) -> str | None:
            return {
                "Content-Encoding": "identity",
                "Content-Length": "2",
                "Content-Type": "application/json; charset=utf-8",
            }.get(name, default)

        @staticmethod
        def read(amount: int) -> bytes:
            captured["read_amount"] = amount
            return b"{}"

    class _Connection:
        def __init__(self, host: str, port: int, *, timeout: float) -> None:
            captured["connection"] = (host, port, timeout)

        def request(
            self,
            method: str,
            path: str,
            *,
            body: bytes | None,
            headers: dict[str, str],
        ) -> None:
            captured["request"] = (method, path, body, headers)

        @staticmethod
        def getresponse() -> _Response:
            return _Response()

        @staticmethod
        def close() -> None:
            captured["closed"] = True

    monkeypatch.setattr(
        regional_ground_authority_transport, "HTTPConnection", _Connection
    )
    runtime = AuthorityRuntime(process=object(), token=token, port=43123)  # type: ignore[arg-type]
    transport = LoopbackAuthorityHttpTransport(runtime, timeout_s=1.25)

    response = transport.request("POST", JOB_COLLECTION_PATH, b"{}", 8)

    assert captured["connection"] == (LOOPBACK_HOST, 43123, 1.25)
    method, path, body, headers = captured["request"]  # type: ignore[misc]
    assert (method, path, body) == ("POST", JOB_COLLECTION_PATH, b"{}")
    assert headers["Authorization"] == f"Bearer {token}"
    assert headers["Cache-Control"] == "no-store"
    assert captured["read_amount"] == 9
    assert captured["closed"] is True
    assert response == AuthorityHttpResponse(200, "application/json", b"{}")


def test_closed_transport_rejects_without_opening_a_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        regional_ground_authority_transport,
        "HTTPConnection",
        lambda *_args, **_kwargs: pytest.fail("connection must remain closed"),
    )
    runtime = AuthorityRuntime(process=object(), token="secret", port=43123)  # type: ignore[arg-type]
    transport = LoopbackAuthorityHttpTransport(runtime, timeout_s=1.0)
    transport.close()

    with pytest.raises(RuntimeError, match="transport is closed"):
        transport.request("GET", JOB_COLLECTION_PATH, None, 8)


@pytest.mark.parametrize(
    "path",
    [
        "/api/rate-of-closure/v1/../secret",
        "/api/rate-of-closure/v1/jobs#fragment",
        "/api/rate-of-closure/v1/jobs\\cancel",
        "/api/rate-of-closure/v1/jobs\nother",
    ],
)
def test_transport_rejects_ambiguous_or_control_character_paths(path: str) -> None:
    runtime = AuthorityRuntime(process=object(), token="secret", port=43123)  # type: ignore[arg-type]
    transport = LoopbackAuthorityHttpTransport(runtime, timeout_s=1.0)

    with pytest.raises(ValueError, match="outside the fixed API root"):
        transport.request("GET", path, None, 8)
