"""GUI registration for Folder Tool."""

from __future__ import annotations

GUI_INFO = {
    "name": "Folder Tool (Utility)",
    "tool_name": "folder_tool",
    "description": "Directory Management Utility",
    "category": "Development Tools",
    "icon": "folder",
    "pyqt6": {
        "module": "Folders_Tool_r0",
        "class": "FolderTool",
        "dependencies": ["tkinter"],
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
