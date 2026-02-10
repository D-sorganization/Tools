"""GUI registration for WGS Reactor Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "WGS Reactor Calculator",
    "tool_name": "wgs_reactor",
    "description": "Water-Gas Shift reactor equilibrium and sizing calculations",
    "category": "Process Simulation",
    "icon": "reactor",
    "pyqt6": {
        "module": "wgs_reactor.ui.pyqt6.main_window",
        "class": "WGSReactorMainWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "WGSReactor",
    },
    "web": {
        "port": 5178,
        "auto_open_browser": False,
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
