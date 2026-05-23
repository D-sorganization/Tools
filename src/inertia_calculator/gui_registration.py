"""GUI registration for Inertia Calculator."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Inertia Calculator",
    "tool_name": "inertia_calculator",
    "description": "Calculate and validate inertia tensors for rigid bodies",
    "category": "robotics",
    "icon": "cube",
    "entry_point": "inertia_calculator.launch_pyqt6:main",
    "pyqt6": {
        "module": "inertia_calculator.ui.pyqt6.main_window",
        "class": "InertiaCalculatorWindow",
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "InertiaCalculator",
    },
}

# Backward-compatible alias expected by tests and legacy consumers.
GUI_METADATA = GUI_INFO


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
