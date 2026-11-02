"""Import helpers for flat-layout P1AM backend tests (issue #5924).

The backend lives at ``src/p1am_control_system/backend`` and exposes ``main`` on
``sys.path`` (see ``tests/p1am_control_system/conftest.py`` and
``pyproject.toml`` ``pythonpath``). Only missing optional runtime deps may
skip; unexpected import failures must propagate.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

_BACKEND_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "p1am_control_system" / "backend"
)
_OPTIONAL_RUNTIME = ("fastapi", "sqlmodel")


def ensure_backend_on_path() -> None:
    backend_str = str(_BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


def load_backend_main() -> ModuleType:
    """Import ``main`` after explicit optional-runtime prerequisites."""
    for module_name in _OPTIONAL_RUNTIME:
        pytest.importorskip(module_name)
    ensure_backend_on_path()
    return importlib.import_module("main")
