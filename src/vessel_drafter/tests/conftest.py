"""Conftest for vessel drafter tests -- ensures package is importable."""

import sys
from pathlib import Path

_PYTHON_DIR = str(Path(__file__).resolve().parent.parent / "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)
