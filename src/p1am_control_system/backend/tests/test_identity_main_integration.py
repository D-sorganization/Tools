"""Application composition contract for the named identity surface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PLC_DRIVER", "modbus")
sys.path.insert(0, str(Path(__file__).parent.parent))

from _route_inventory import methods_by_path as _methods_by_path  # noqa: E402
from audit_middleware import MutationAuditMiddleware  # noqa: E402
from fastapi import APIRouter, FastAPI  # noqa: E402
from main import _configuration_revision, app, configuration_workflow  # noqa: E402


def test_route_inventory_resolves_routes_behind_an_included_router() -> None:
    """The inventory must see through `include_router`, not around it.

    From FastAPI 0.141 an included router appears in ``app.routes`` as a single
    pathless ``_IncludedRouter`` marker that does not expose its children. An
    inventory that skipped such markers returned an empty mapping, which turned
    every downstream assertion into a vacuous pass.
    """
    router = APIRouter()

    @router.post("/session")
    def _create() -> dict[str, str]:
        return {}

    probe = FastAPI()
    probe.include_router(router, prefix="/api/auth")

    assert _methods_by_path(probe) == {"/api/auth/session": {"POST"}}


def test_main_application_mounts_identity_session_routes() -> None:
    methods_by_path = _methods_by_path(app)

    assert methods_by_path, "route inventory empty; assertions below are vacuous"

    assert "POST" in methods_by_path["/api/auth/session"]
    assert "DELETE" in methods_by_path["/api/auth/session"]
    assert "GET" in methods_by_path["/api/auth/me"]
    assert "GET" in methods_by_path["/api/audit"]
    assert "GET" in methods_by_path["/api/alarm-management/active"]
    assert "POST" in methods_by_path["/api/alarm-management/{tag}/shelf"]
    assert "POST" in methods_by_path["/api/configurations/drafts"]
    assert "POST" in methods_by_path["/api/configurations/{revision_id}/activate"]
    assert "GET" in methods_by_path["/api/system/identity"]
    assert "GET" in methods_by_path["/api/system/health"]
    assert "POST" in methods_by_path["/api/system/backups"]
    assert "POST" in methods_by_path["/api/system/restores"]
    assert "GET" in methods_by_path["/api/acceptance/scenarios/representative"]
    assert "POST" in methods_by_path["/api/acceptance/scenarios/run"]


def test_main_application_registers_automatic_mutation_audit() -> None:
    assert any(
        middleware.cls is MutationAuditMiddleware for middleware in app.user_middleware
    )


def test_audit_revision_resolves_the_identified_active_configuration(
    monkeypatch,
) -> None:
    active = type("ActiveRevision", (), {"activation_identity": "cfg-000042-proof"})()
    monkeypatch.setattr(configuration_workflow, "active", lambda: active)

    assert _configuration_revision() == "cfg-000042-proof"
