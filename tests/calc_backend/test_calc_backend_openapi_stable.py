"""OpenAPI stability guard for calc_backend (issue #6020)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("calc_backend")

from calc_backend.app import app

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_BASELINE = _FIXTURES / "calc_backend_openapi_baseline.v1.json"


def test_openapi_document_byte_identical_to_baseline() -> None:
    """Registry refactors must not change the published OpenAPI bytes."""
    app.openapi_schema = None
    generated = json.dumps(app.openapi(), sort_keys=True, separators=(",", ":"))
    baseline = _BASELINE.read_text(encoding="utf-8")
    assert generated == baseline
