"""Tests for the state-change request guard (CSRF / cross-origin) — #4037.

The Pi runs a kiosk Chromium against the HMI, and several control routes take
**no request body** (``POST /api/estop``, ``/api/estop/clear``,
``/api/power_supply/acknowledge_trip``, ``/api/temperature/acknowledge_trip``,
``/api/pid/{i}/tuning/start``). Those are CORS-"simple" requests: any page the
operator happens to open can issue ``fetch(url, {method: "POST",
mode: "no-cors"})`` and the browser will hide the *response* but not the
*effect* — a real 110 V heater or DC supply command.

Two independent controls close that hole, both enforced before routing:

1. **Origin allowlist.** A cross-site ``fetch``/form POST always carries an
   ``Origin`` header. If one is present and not in the CORS allowlist the
   request is refused. A credentialed ``curl`` sends no ``Origin`` and is
   unaffected.
2. **Preflight forcing.** Every other state-changing request must carry a
   non-simple signal — a custom header (``X-Requested-With`` or the API-key
   header) or ``Content-Type: application/json``. None of those can be produced
   by a simple request, so the browser is forced into a preflight the guard's
   origin allowlist then answers for.

``POST /api/estop`` is deliberately exempt from (2) only: a panic stop must stay
reachable from a bare shell script. It is *not* exempt from (1), so a malicious
page still cannot trip the plant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).parent.parent))

from cors_config import (  # noqa: E402
    CSRF_HEADER_NAME,
    CSRF_HEADER_VALUE,
    PREFLIGHT_EXEMPT_PATHS,
    STATE_CHANGING_METHODS,
    RequestGuardMiddleware,
    evaluate_state_change,
)
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_ALLOWED = ("http://localhost:3002", "http://127.0.0.1:3002")
_JSON = {"content-type": "application/json"}


# --------------------------------------------------------------------------- #
# Pure decision function                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", sorted(STATE_CHANGING_METHODS))
def test_simple_cross_origin_post_is_refused(method: str) -> None:
    reason = evaluate_state_change(
        method=method,
        path="/api/tags/TAG_0",
        headers={"origin": "http://evil.example"},
        allowed_origins=_ALLOWED,
    )
    assert reason is not None


def test_same_origin_json_request_is_allowed() -> None:
    assert (
        evaluate_state_change(
            method="POST",
            path="/api/tags/TAG_0",
            headers={"origin": "http://localhost:3002", **_JSON},
            allowed_origins=_ALLOWED,
        )
        is None
    )


def test_bodyless_post_without_custom_header_is_refused() -> None:
    """The CSRF-able shape from #4037: no body, no header, no preflight."""
    assert (
        evaluate_state_change(
            method="POST",
            path="/api/estop/clear",
            headers={},
            allowed_origins=_ALLOWED,
        )
        is not None
    )


def test_bodyless_post_with_custom_header_is_allowed() -> None:
    assert (
        evaluate_state_change(
            method="POST",
            path="/api/estop/clear",
            headers={CSRF_HEADER_NAME.lower(): CSRF_HEADER_VALUE},
            allowed_origins=_ALLOWED,
        )
        is None
    )


def test_api_key_header_also_forces_a_preflight() -> None:
    """``X-API-Key`` is itself a custom header — it cannot be sent simply."""
    assert (
        evaluate_state_change(
            method="POST",
            path="/api/estop/clear",
            headers={"x-api-key": "whatever"},  # pragma: allowlist secret
            allowed_origins=_ALLOWED,
        )
        is None
    )


def test_text_plain_json_smuggling_is_refused() -> None:
    """A simple request can send ``text/plain`` with a JSON body; FastAPI would
    still parse it. The guard must reject it."""
    assert (
        evaluate_state_change(
            method="POST",
            path="/api/tags/TAG_0",
            headers={"content-type": "text/plain"},
            allowed_origins=_ALLOWED,
        )
        is not None
    )


def test_form_encoded_body_is_refused() -> None:
    assert (
        evaluate_state_change(
            method="POST",
            path="/api/tags/TAG_0",
            headers={"content-type": "application/x-www-form-urlencoded"},
            allowed_origins=_ALLOWED,
        )
        is not None
    )


def test_reads_are_never_blocked() -> None:
    for method in ("GET", "HEAD", "OPTIONS"):
        assert (
            evaluate_state_change(
                method=method,
                path="/api/routing",
                headers={"origin": "http://evil.example"},
                allowed_origins=_ALLOWED,
            )
            is None
        )


def test_panic_stop_is_exempt_from_preflight_forcing_only() -> None:
    """A bare ``curl -X POST /api/estop`` must keep working…"""
    assert "/api/estop" in PREFLIGHT_EXEMPT_PATHS
    assert (
        evaluate_state_change(
            method="POST",
            path="/api/estop",
            headers={},
            allowed_origins=_ALLOWED,
        )
        is None
    )


def test_panic_stop_still_refuses_a_foreign_origin() -> None:
    """…but a malicious page still cannot trip the plant."""
    assert (
        evaluate_state_change(
            method="POST",
            path="/api/estop",
            headers={"origin": "http://evil.example"},
            allowed_origins=_ALLOWED,
        )
        is not None
    )


def test_empty_allowlist_refuses_every_origin() -> None:
    """Fail closed: no allowlist means no browser origin is trusted."""
    assert (
        evaluate_state_change(
            method="POST",
            path="/api/estop",
            headers={"origin": "http://localhost:3002"},
            allowed_origins=(),
        )
        is not None
    )


# --------------------------------------------------------------------------- #
# ASGI middleware wiring                                                       #
# --------------------------------------------------------------------------- #


def _guarded_app() -> FastAPI:
    app = FastAPI()

    @app.post("/api/estop")
    async def estop() -> dict[str, str]:
        return {"status": "tripped"}

    @app.post("/api/estop/clear")
    async def clear() -> dict[str, str]:
        return {"status": "cleared"}

    @app.get("/api/routing")
    async def routing() -> dict[str, str]:
        return {"status": "ok"}

    app.add_middleware(RequestGuardMiddleware, allowed_origins=_ALLOWED)
    return app


def test_middleware_blocks_foreign_origin_post() -> None:
    client = TestClient(_guarded_app())
    resp = client.post("/api/estop", headers={"Origin": "http://evil.example"})
    assert resp.status_code == 403


def test_middleware_blocks_simple_bodyless_post() -> None:
    client = TestClient(_guarded_app())
    resp = client.post("/api/estop/clear", headers={"Content-Type": "text/plain"})
    assert resp.status_code == 403


def test_middleware_allows_hmi_shaped_post() -> None:
    client = TestClient(_guarded_app())
    resp = client.post(
        "/api/estop/clear",
        headers={
            "Origin": "http://localhost:3002",
            CSRF_HEADER_NAME: CSRF_HEADER_VALUE,
        },
    )
    assert resp.status_code == 200


def test_middleware_allows_reads() -> None:
    client = TestClient(_guarded_app())
    assert client.get("/api/routing").status_code == 200


def test_middleware_denial_is_json() -> None:
    client = TestClient(_guarded_app())
    resp = client.post("/api/estop", headers={"Origin": "http://evil.example"})
    assert resp.headers["content-type"].startswith("application/json")
    assert "detail" in resp.json()
