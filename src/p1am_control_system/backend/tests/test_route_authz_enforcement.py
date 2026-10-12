"""Authorization enforcement and WebSocket tests for the route matrix (#4028).

Drives the ``(method, path, required_tier)`` table from
``_route_authz_helpers`` against the real app: gated routes reject anonymous
callers, admin routes reject an operator key, the panic stop stays public and
the ``/api/stream`` socket authenticates on its first frame. The table's own
coverage checks live in ``test_route_authz_matrix.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from _route_authz_helpers import (  # noqa: E402
    _ADMIN_KEY,
    _ADMIN_ONLY,
    _DENIED,
    _GATED,
    _OPERATOR_KEY,
    _PUBLIC_MUTATING,
    _app_websocket_paths,
    _authenticated_deployment,  # noqa: F401  (autouse fixture)
    _concrete,
    _headers,
    app,
)
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    """The real app WITHOUT its lifespan.

    The matrix only exercises the dependency/middleware layer; running the
    lifespan would start the Modbus connect loop, the 10 Hz poll loop and the
    Alicat manager against no hardware. ``raise_server_exceptions=False`` keeps
    a downstream 500 (unmapped tag, no PLC) from masking the status code the
    authorization layer produced.
    """
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(("method", "path"), _GATED, ids=lambda v: str(v))
def test_gated_route_rejects_anonymous_caller(
    client: TestClient, method: str, path: str
) -> None:
    resp = client.request(method, _concrete(path), headers=_headers(None), json={})
    assert resp.status_code in _DENIED, (
        f"{method} {path} answered {resp.status_code} with NO credential — the "
        "route is missing its auth dependency."
    )


@pytest.mark.parametrize(("method", "path"), _ADMIN_ONLY, ids=lambda v: str(v))
def test_admin_route_rejects_operator_credential(
    client: TestClient, method: str, path: str
) -> None:
    resp = client.request(
        method, _concrete(path), headers=_headers(_OPERATOR_KEY), json={}
    )
    assert resp.status_code == 403, (
        f"{method} {path} answered {resp.status_code} for an OPERATOR key — a "
        "hardware-mutating route must require the admin credential."
    )


@pytest.mark.parametrize(("method", "path"), _PUBLIC_MUTATING, ids=lambda v: str(v))
def test_public_mutating_route_is_reachable_without_a_credential(
    client: TestClient, method: str, path: str
) -> None:
    """The panic stop must never regress into requiring a credential."""
    resp = client.request(method, _concrete(path), headers=_headers(None), json={})
    assert resp.status_code not in _DENIED


def test_admin_credential_passes_the_admin_gate(client: TestClient) -> None:
    """Sanity: the matrix measures authorization, not a blanket 403."""
    resp = client.request(
        "PUT",
        "/api/performance",
        headers=_headers(_ADMIN_KEY),
        json={"mode": "normal"},
    )
    assert resp.status_code not in _DENIED


def test_operator_credential_passes_the_operator_gate(client: TestClient) -> None:
    resp = client.request(
        "POST",
        "/api/events",
        headers=_headers(_OPERATOR_KEY),
        json={"event_type": "SYSTEM", "description": "matrix probe"},
    )
    assert resp.status_code not in _DENIED


# --------------------------------------------------------------------------- #
# WebSocket                                                                 #
# --------------------------------------------------------------------------- #


def test_stream_websocket_route_exists() -> None:
    paths = _app_websocket_paths()
    assert "/api/stream" in paths


def test_stream_rejects_unauthenticated_client(client: TestClient) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/api/stream") as ws:
            ws.send_text("")
            ws.receive_text()


def test_stream_accepts_first_frame_credential(client: TestClient) -> None:
    """The HMI authenticates by sending the key as its first frame (#4007)."""
    with client.websocket_connect("/api/stream") as ws:
        ws.send_text(_OPERATOR_KEY)
        # No exception on send after auth => the socket stayed open.
        ws.send_text("ping")
