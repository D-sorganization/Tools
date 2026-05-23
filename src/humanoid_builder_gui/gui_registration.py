# ruff: noqa: E501
"""GUI registration for Humanoid Character Builder."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Humanoid Character Builder",
    "tool_name": "humanoid_builder_gui",
    "description": "Build parametric humanoid characters with anthropometric calculations",
    "category": "robotics",
    "icon": "person",
    "entry_point": "humanoid_builder_gui.launch_pyqt6:main",
    "pyqt6": {
        "module": "humanoid_builder_gui.ui.pyqt6.main_window",
        "class": "HumanoidBuilderWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "HumanoidBuilder",
    },
}

# Backward-compatible alias expected by tests and legacy consumers.
GUI_METADATA = GUI_INFO


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
