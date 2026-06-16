"""GUI registration for Inertia Calculator."""

from __future__ import annotations

from typing import Any

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

GUI_METADATA = {
    "name": GUI_INFO["name"],
    "description": GUI_INFO["description"],
    "category": GUI_INFO["category"].lower(),
    "entry_point": GUI_INFO["pyqt6"]["module"],
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
