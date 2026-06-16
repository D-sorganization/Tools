"""Root conftest for pendulum_simulator.

Ensures local and shared source roots are on sys.path so package-local pytest
runs match the monorepo test environment without requiring editable installs.
"""

from __future__ import annotations

import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_SRC_DIR = str(_THIS_FILE.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_SHARED_DIR = str(_THIS_FILE.parents[1] / "shared" / "python")
if _SHARED_DIR not in sys.path:
    sys.path.insert(0, _SHARED_DIR)
