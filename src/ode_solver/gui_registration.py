"""
ODE Solver - GUI Registration
=============================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "ODE Solver",
    "description": "Solve systems of ordinary differential equations symbolically",
    "category": "mathematics",
    "version": "1.0.0",
    "entry_point": "ode_solver.ui.pyqt6.main_window:ODESolverWindow",
    "icon": "function",
    "keywords": [
        "ODE",
        "differential equations",
        "symbolic",
        "numerical",
        "solver",
        "scipy",
        "sympy",
    ],
    "dependencies": {
        "required": ["PyQt6", "numpy", "scipy", "sympy"],
        "optional": ["matplotlib"],
    },
    "features": [
        "Symbolic ODE definition",
        "Multi-variable systems",
        "Configurable parameters",
        "Time series solution",
        "Initial condition support",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
