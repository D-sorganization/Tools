"""GUI registration for the P&ID Generator tool."""

from __future__ import annotations

GUI_INFO = {
    "name": "P&ID Generator",
    "tool_name": "pid_generator",
    "description": "Generate P&ID drawings from YAML specifications (DXF + SVG output)",
    "category": "Engineering Drafting",
    "icon": "drafting",
    "pyqt6": {
        "module": "pid_generator.launch_pyqt6",
        "dependencies": ["ezdxf"],
        "settings_app": "PIDGenerator",
        "min_size": [800, 600],
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
