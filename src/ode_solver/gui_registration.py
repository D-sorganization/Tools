"""GUI registration for ODE Solver."""

from __future__ import annotations

from typing import Any

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
    "web": {
        "path": "web",
        "port": 5174,
        "auto_open_browser": True,
    },
}

GUI_METADATA = {
    "name": GUI_INFO["name"],
    "description": GUI_INFO["description"],
    "category": "mathematics",
    "entry_point": "ode_solver.launch_pyqt6:main",
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
