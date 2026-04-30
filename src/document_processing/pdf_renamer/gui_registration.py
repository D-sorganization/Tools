"""GUI registration for PDF Renamer."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "PDF Renamer",
    "tool_name": "pdf_renamer",
    "description": "Intelligent PDF File Renaming Tool",
    "category": "Development Tools",
    "icon": "pdf",
    "pyqt6": {
        "module": "pdf_renamer.gui",
        "class": "PDFRenamer",
        "dependencies": ["PyQt6", "pdfplumber"],
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
