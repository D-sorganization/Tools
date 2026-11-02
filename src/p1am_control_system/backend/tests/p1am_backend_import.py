"""Shared import guard for backend-local pytest modules."""

from __future__ import annotations

from typing import Any

import pytest


def require_main() -> Any:
    """Return the backend ``main`` module or skip when prerequisites are absent."""
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    try:
        import main
    except ModuleNotFoundError as exc:  # pragma: no cover - optional stack
        pytest.skip(f"P1AM backend not importable in this environment: {exc}")
    return main
