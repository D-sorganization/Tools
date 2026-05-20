"""GUI registration for P1AM HMI Control System Desktop App."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "P1AM HMI Control System",
    "tool_name": "p1am_control_system",
    "description": "HMI Control System for P1AM-100 PLC",
    "category": "Industrial",
    "icon": "desktop",
    "pyqt6": {
        "module": "p1am_control_system.desktop.main_window",
        "class": "P1AMMainWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "P1AMControlSystem",
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
