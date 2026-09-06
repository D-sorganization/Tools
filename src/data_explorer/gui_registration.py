"""GUI registration for the Data Explorer workbench."""

from __future__ import annotations

from typing import Any

GUI_INFO: dict[str, Any] = {
    "name": "Data Explorer",
    "tool_name": "data_explorer",
    "description": "Interactive workbench for browsing simulation datasets",
    "category": "Data Processing",
    "icon": "table",
    "maturity": "stable",
    "help": "src/data_explorer/gui.py",
    "pyqt6": {
        "module": "data_explorer.gui",
        "class": "DataExplorerWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "DataExplorer",
    },
    "web": False,
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
