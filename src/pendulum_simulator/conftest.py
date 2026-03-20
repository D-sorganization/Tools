"""Root conftest for pendulum_simulator.

Ensures 'src/' is on sys.path so that ``from double_pendulum_golf import ...``
works when running pytest from the repository root (without editable install).
"""

from __future__ import annotations

import pathlib
import sys

_THIS_FILE = pathlib.Path(__file__).resolve()
_SRC_DIR = str(_THIS_FILE.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
