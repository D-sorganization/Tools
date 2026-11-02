"""Import helpers for P1AM backend tests (package/lane contract)."""

from __future__ import annotations

from typing import Any

import pytest


def require_backend_main() -> Any:
    """Import ``main`` from the backend directory on ``sys.path``.

    Skips only when optional runtime dependencies or ``main`` itself are missing.
    Other import failures propagate so backend defects fail loudly.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("sqlmodel")
    try:
        import main
    except ModuleNotFoundError as exc:
        pytest.skip(f"P1AM backend not importable in this environment: {exc}")
    return main
