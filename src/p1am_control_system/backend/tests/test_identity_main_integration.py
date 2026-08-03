"""Application composition contract for the named identity surface."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PLC_DRIVER", "modbus")
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_middleware import MutationAuditMiddleware  # noqa: E402
from main import app  # noqa: E402


def test_main_application_mounts_identity_session_routes() -> None:
    methods_by_path: dict[str, set[str]] = {}
    for route in app.routes:
        methods_by_path.setdefault(route.path, set()).update(
            getattr(route, "methods", set())
        )

    assert "POST" in methods_by_path["/api/auth/session"]
    assert "DELETE" in methods_by_path["/api/auth/session"]
    assert "GET" in methods_by_path["/api/auth/me"]
    assert "GET" in methods_by_path["/api/audit"]
    assert "GET" in methods_by_path["/api/alarm-management/active"]
    assert "POST" in methods_by_path["/api/alarm-management/{tag}/shelf"]


def test_main_application_registers_automatic_mutation_audit() -> None:
    assert any(
        middleware.cls is MutationAuditMiddleware for middleware in app.user_middleware
    )
