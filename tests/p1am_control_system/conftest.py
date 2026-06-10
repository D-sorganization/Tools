"""Test fixtures for P1AM control-system tests.

The backend package uses a flat ``sys.path`` import style (``from main import
app``) rather than package-qualified imports. Make the backend directory
importable so the backend security regression tests (issue #3289/#3292) can be
collected and run under the repo-wide ``tests/`` tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = (
    Path(__file__).resolve().parents[2] / "src" / "p1am_control_system" / "backend"
)

if _BACKEND_DIR.is_dir():
    backend_str = str(_BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)
