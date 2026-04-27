"""Test configuration for ode_solver UI tests.

Adds the ode_solver Python package root to sys.path so that
``from ode_solver.ui.pyqt6.main_window import ...`` resolves correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# src/ode_solver/python contains the top-level 'ode_solver' package
_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "ode_solver" / "python"
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
