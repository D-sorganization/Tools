"""GUI registration for Folder Packer Pro."""

from __future__ import annotations

GUI_INFO = {
    "name": "Folder Packer Pro",
    "tool_name": "folder_packer_pro",
    "description": "Professional Project Archiving and Distribution Tool",
    "category": "Development Tools",
    "icon": "package",
    "pyqt6": {
        "module": "folder_packer_pro",
        "class": "FolderPackerPro",
        "dependencies": ["tkinter", "cryptography"],
    },
}

def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
