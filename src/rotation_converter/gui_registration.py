"""GUI registration for Rotation Converter."""

from __future__ import annotations

GUI_INFO = {
    "name": "Rotation Converter",
    "tool_name": "rotation_converter",
    "description": (
        "Comprehensive rotation and rigid-body transform converter with "
        "interactive 3D visualization. Supports quaternions, Euler angles, "
        "rotation matrices, axis-angle, SE(3), twists, screw axes, and "
        "frame-aware transforms."
    ),
    "category": "Robotics",
    "icon": "rotate",
    "pyqt6": {
        "module": "rotation_converter.ui.pyqt6.main_window",
        "class": "RotationConverterMainWindow",
        "dependencies": ["PyQt6", "matplotlib"],
        "settings_app": "RotationConverter",
        "min_size": [1200, 800],
    },
    "web": {
        "port": 5192,
        "auto_open_browser": True,
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
