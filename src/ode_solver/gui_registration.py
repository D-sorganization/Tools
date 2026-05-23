"""GUI registration for ODE Solver."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "ODE Solver",
    "tool_name": "ode_solver",
    "description": "Solve systems of ordinary differential equations symbolically",
    "category": "mathematics",
    "icon": "function",
    "entry_point": "ode_solver.launch_pyqt6:main",
    "pyqt6": {
        "module": "ode_solver.ui.pyqt6.main_window",
        "class": "ODESolverWindow",
        "dependencies": ["PyQt6", "numpy", "scipy", "sympy"],
        "settings_app": "ODESolver",
    },
}

# Backward-compatible alias expected by tests and legacy consumers.
GUI_METADATA = GUI_INFO


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
