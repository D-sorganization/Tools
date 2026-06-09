"""GUI registration for Movement Optimizer."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Movement Optimizer",
    "tool_name": "optimizer_gui",
    "description": (
        "Optimize motion policies with Adam, swingset, and chain dynamics models"
    ),
    "category": "Optimization",
    "icon": "chart-line",
    "pyqt6": {
        "module": "optimizer_gui.ui.pyqt6.main_window",
        "class": "OptimizerWindow",
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "OptimizerGui",
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
