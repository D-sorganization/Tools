"""Tests for the read-authentication gate (``P1AM_REQUIRE_READ_AUTH``).

The read surface (``/api/routing``, historian trends/export, snapshot, events,
plant tree, ladder explorer, the service ``/config`` + ``/status`` pairs and the
whole ``/api/explorer`` router) is credential-gated **by default** as of issue
#4037: ``GET /api/routing`` alone discloses the full register map, every scale
factor and every interlock trip limit. ``P1AM_REQUIRE_READ_AUTH=0`` (or
``P1AM_DEV_NO_AUTH=1``) opts a bench setup back out.

These tests build tiny FastAPI apps around the real dependency and the explorer
router (mirroring ``tests/test_data_explorer_router.py``) rather than booting the
full backend, so the gate is exercised in isolation. Both directions are pinned
explicitly: secure when unset, public only when opted out.

Every variable the gate reads is set to an explicit value per test rather than
left unset, so the outcome cannot depend on which sibling module imported first
(#4061).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlmodel")

# ``main`` builds a process-wide PLC client at import time from the (lru-cached)
# settings, and that module object is shared across the whole test session. The
# functional suite (``tests/test_backend.py``) requires the modbus driver so it
# can patch ``modbus_manager.client``; select the same driver here so this
# module importing ``main`` first cannot leave the shared client as a simulator
# and break that suite (collection order is not guaranteed). We do NOT enable
# any auth env var — the read-auth gate is toggled per-test below.
os.environ.setdefault("PLC_DRIVER", "modbus")

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth_config import (
    CREDENTIAL_HEADER_NAME,  # noqa: E402
    require_read_auth,  # noqa: E402
)
from data_explorer_router import create_data_explorer_router  # noqa: E402
from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

_TEST_KEY = "read-surface-secret"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start each test from a known-clean auth environment.

    ``require_read_auth`` and ``require_api_key`` read process env at request
    time, so leaked vars from sibling suites would otherwise change behavior.
    """
    for var in ("P1AM_REQUIRE_READ_AUTH", "P1AM_DEV_NO_AUTH", "P1AM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield


def _read_app() -> FastAPI:
    """A minimal app whose single read route carries the opt-in gate."""
    app = FastAPI()

    @app.get("/api/snapshot", dependencies=[Depends(require_read_auth)])
    def snapshot() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _explorer_app() -> FastAPI:
    """An app mounting the explorer router gated by the read-auth dependency."""

    def get_session() -> Iterator[None]:
        yield None

    app = FastAPI()
    app.include_router(
        create_data_explorer_router(get_session, read_auth_dep=require_read_auth)
    )
    return app


# --------------------------------------------------------------------------- #
# Default ON: the read surface is gated unless explicitly opted out (#4037)    #
# --------------------------------------------------------------------------- #


def test_read_route_gated_when_setting_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Secure by default: an unset variable must NOT mean a public API."""
    monkeypatch.setenv("P1AM_API_KEY", _TEST_KEY)
    resp = TestClient(_read_app()).get("/api/snapshot")
    assert resp.status_code in (401, 403)


def test_read_route_public_when_setting_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bench opt-out still works, but it now has to be asked for."""
    monkeypatch.setenv("P1AM_REQUIRE_READ_AUTH", "false")
    resp = TestClient(_read_app()).get("/api/snapshot")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_explorer_route_public_when_setting_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_REQUIRE_READ_AUTH", "false")
    resp = TestClient(_explorer_app()).post(
        "/api/explorer/statistics",
        json={"columns": [{"name": "a", "values": [1.0, 2.0, 3.0]}]},
    )
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# ON: gate enforces the operator key (DEV_NO_AUTH off)                         #
# --------------------------------------------------------------------------- #


def test_read_route_rejects_without_key_when_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_REQUIRE_READ_AUTH", "1")
    monkeypatch.setenv("P1AM_API_KEY", _TEST_KEY)
    resp = TestClient(_read_app()).get("/api/snapshot")
    assert resp.status_code in (401, 403)


def test_read_route_accepts_correct_key_when_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_REQUIRE_READ_AUTH", "1")
    monkeypatch.setenv("P1AM_API_KEY", _TEST_KEY)
    resp = TestClient(_read_app()).get(
        "/api/snapshot", headers={CREDENTIAL_HEADER_NAME: _TEST_KEY}
    )
    assert resp.status_code == 200


def test_read_route_rejects_wrong_key_when_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_REQUIRE_READ_AUTH", "1")
    monkeypatch.setenv("P1AM_API_KEY", _TEST_KEY)
    resp = TestClient(_read_app()).get(
        "/api/snapshot", headers={CREDENTIAL_HEADER_NAME: "wrong"}
    )
    assert resp.status_code in (401, 403)


def test_explorer_route_rejects_without_key_when_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_REQUIRE_READ_AUTH", "1")
    monkeypatch.setenv("P1AM_API_KEY", _TEST_KEY)
    resp = TestClient(_explorer_app()).post(
        "/api/explorer/statistics",
        json={"columns": [{"name": "a", "values": [1.0, 2.0, 3.0]}]},
    )
    assert resp.status_code in (401, 403)


def test_explorer_route_accepts_correct_key_when_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("P1AM_REQUIRE_READ_AUTH", "1")
    monkeypatch.setenv("P1AM_API_KEY", _TEST_KEY)
    resp = TestClient(_explorer_app()).post(
        "/api/explorer/statistics",
        json={"columns": [{"name": "a", "values": [1.0, 2.0, 3.0]}]},
        headers={CREDENTIAL_HEADER_NAME: _TEST_KEY},
    )
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# DEV_NO_AUTH bypass still wins even with the gate on                          #
# --------------------------------------------------------------------------- #


def test_dev_no_auth_bypasses_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P1AM_REQUIRE_READ_AUTH", "1")
    monkeypatch.setenv("P1AM_DEV_NO_AUTH", "1")
    resp = TestClient(_read_app()).get("/api/snapshot")
    assert resp.status_code == 200


def test_setting_default_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    from settings import P1AMSettings

    monkeypatch.delenv("P1AM_REQUIRE_READ_AUTH", raising=False)
    assert P1AMSettings(_env_file=None).require_read_auth is True
