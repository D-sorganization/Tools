"""GUI registration for Parametric URDF Builder."""

from __future__ import annotations

GUI_INFO = {
    "name": "Parametric URDF Builder",
    "tool_name": "urdf_builder_gui",
    "description": "Generate parametric URDF models for robotics applications",
    "category": "Robotics",
    "icon": "robot",
    "pyqt6": {
        "module": "urdf_builder_gui.ui.pyqt6.main_window",
        "class": "URDFBuilderWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "URDFBuilder",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
