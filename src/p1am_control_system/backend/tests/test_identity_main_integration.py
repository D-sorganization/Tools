"""Application composition contract for the named identity surface."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable
from pathlib import Path

os.environ.setdefault("PLC_DRIVER", "modbus")
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_middleware import MutationAuditMiddleware  # noqa: E402
from main import _configuration_revision, app, configuration_workflow  # noqa: E402


def _methods_by_path(routes: Iterable[object]) -> dict[str, set[str]]:
    methods_by_path: dict[str, set[str]] = {}
    for route in routes:
        path = getattr(route, "path", None)
        if not isinstance(path, str):
            continue
        methods_by_path.setdefault(path, set()).update(getattr(route, "methods", set()))
    return methods_by_path


def test_route_inventory_ignores_optional_pathless_router_markers() -> None:
    assert _methods_by_path((object(),)) == {}


def test_main_application_mounts_identity_session_routes() -> None:
    methods_by_path = _methods_by_path(app.routes)

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
