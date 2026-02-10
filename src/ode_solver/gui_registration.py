"""GUI registration for ODE Solver."""

from __future__ import annotations

GUI_INFO = {
    "name": "ODE Solver",
    "tool_name": "ode_solver",
    "description": "Solve systems of ordinary differential equations symbolically",
    "category": "Mathematics",
    "icon": "function",
    "pyqt6": {
        "module": "ode_solver.ui.pyqt6.main_window",
        "class": "ODESolverWindow",
        "dependencies": ["PyQt6", "numpy", "scipy", "sympy"],
        "settings_app": "ODESolver",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
