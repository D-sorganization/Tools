"""GUI registration for Adam Optimizer."""

from __future__ import annotations

GUI_INFO = {
    "name": "Adam Optimizer",
    "tool_name": "optimizer_gui",
    "description": "Configure and run Adam-based optimization",
    "category": "Optimization",
    "icon": "chart-line",
    "pyqt6": {
        "module": "optimizer_gui.ui.pyqt6.main_window",
        "class": "OptimizerWindow",
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "OptimizerGui",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
