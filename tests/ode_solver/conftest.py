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

_PACKAGE_DIR = _PACKAGE_ROOT / "ode_solver"
_loaded_ode_solver = sys.modules.get("ode_solver")
if _loaded_ode_solver is not None:
    loaded_paths = {str(path) for path in getattr(_loaded_ode_solver, "__path__", [])}
    if str(_PACKAGE_DIR) not in loaded_paths:
        for module_name in list(sys.modules):
            if module_name == "ode_solver" or module_name.startswith("ode_solver."):
                sys.modules.pop(module_name, None)
