"""GUI registration for Inertia Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Inertia Calculator",
    "tool_name": "inertia_calculator",
    "description": "Calculate and validate inertia tensors for rigid bodies",
    "category": "Robotics",
    "icon": "cube",
    "pyqt6": {
        "module": "inertia_calculator.ui.pyqt6.main_window",
        "class": "InertiaCalculatorWindow",
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "InertiaCalculator",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
