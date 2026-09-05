"""GUI registration for the Pendulum Simulator (PyQt6 app + Vite web twin)."""

from __future__ import annotations

from typing import Any

GUI_INFO: dict[str, Any] = {
    "name": "Pendulum Simulator",
    "tool_name": "pendulum_simulator",
    "description": "Multi-link pendulum dynamics with parameter sweeps",
    "category": "Mathematics",
    "icon": "pendulum",
    "maturity": "experimental",
    "help": "src/pendulum_simulator/README.md",
    "pyqt6": {
        "module": "double_pendulum_golf.gui.main_window",
        "class": "MainWindow",
        "dependencies": ["PyQt6", "numpy", "scipy"],
        "settings_app": "PendulumSimulator",
    },
    # pendulum-web/ is a Vite/Tauri twin driven by its own package.json
    # scripts (npm run dev / tauri); it has no Python launch_web.py yet.
    "web": False,
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
