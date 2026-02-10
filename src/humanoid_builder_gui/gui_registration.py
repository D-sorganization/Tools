"""GUI registration for Humanoid Character Builder."""

from __future__ import annotations

GUI_INFO = {
    "name": "Humanoid Character Builder",
    "tool_name": "humanoid_builder_gui",
    "description": "Build parametric humanoid characters with anthropometric calculations",
    "category": "Robotics",
    "icon": "person",
    "pyqt6": {
        "module": "humanoid_builder_gui.ui.pyqt6.main_window",
        "class": "HumanoidBuilderWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "HumanoidBuilder",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
