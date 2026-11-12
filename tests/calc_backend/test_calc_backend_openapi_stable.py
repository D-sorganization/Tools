"""OpenAPI stability guard for calc_backend (issue #6020)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("fastapi")
pytest.importorskip("calc_backend")

from calc_backend.app import app

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_BASELINE = _FIXTURES / "calc_backend_openapi_baseline.v1.json"

_OPERATIONAL_OPENAPI_PATHS: tuple[str, ...] = (
    "/health",
    "/api/health",
    "/api/ready",
    "/api/calc/endpoints",
)


def test_module_openapi_cache_includes_operational_routes() -> None:
    """OpenAPI must be materialized only after all routes are registered."""
    assert app.openapi_schema is not None
    paths = app.openapi_schema.get("paths", {})
    for path in _OPERATIONAL_OPENAPI_PATHS:
        assert path in paths, path


def test_openapi_json_includes_operational_routes() -> None:
    """Served /openapi.json must list health, readiness, and endpoint discovery."""
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    for path in _OPERATIONAL_OPENAPI_PATHS:
        assert path in paths, path


def test_openapi_document_byte_identical_to_baseline() -> None:
    """Registry refactors must not change the published OpenAPI bytes."""
    app.openapi_schema = None
    generated = json.dumps(app.openapi(), sort_keys=True, separators=(",", ":"))
    baseline = _BASELINE.read_text(encoding="utf-8")
    assert generated == baseline
