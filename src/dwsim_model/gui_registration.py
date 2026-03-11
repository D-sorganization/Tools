"""GUI registration for the DWSIM Gasification Model tool."""

from __future__ import annotations

GUI_INFO = {
    "name": "DWSIM Gasification Model",
    "tool_name": "dwsim_model",
    "description": "DWSIM-backed gasification process simulation with Tkinter GUI",
    "category": "Process Simulation",
    "icon": "chemistry",
    "tkinter": {
        "module": "dwsim_model.gui.main_window",
        "launch": "launch",
        "dependencies": ["pythonnet", "pydantic", "PyYAML"],
        "settings_app": "DWSIMModel",
        "min_size": [1200, 800],
    },
    "pyqt6": {
        "enabled": False,
        "note": "GUI currently uses Tkinter; PyQt6 port planned",
    },
    "web": {
        "enabled": False,
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
