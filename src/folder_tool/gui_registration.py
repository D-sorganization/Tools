"""GUI registration for Folder Tool."""

from __future__ import annotations
from typing import Any

GUI_INFO = {
    "name": "Folder Tool (Utility)",
    "tool_name": "folder_tool",
    "description": "Directory Management Utility",
    "category": "Development Tools",
    "icon": "folder",
    # Headless import (Tools #4916): module Folders_Tool_r0 exposes
    # FolderProcessorApp, not FolderTool; promote once the entry class is fixed.
    "maturity": "experimental",
    "pyqt6": {
        "module": "Folders_Tool_r0",
        "class": "FolderTool",
        "dependencies": ["tkinter"],
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
