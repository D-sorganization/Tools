"""GUI registration for C3D Motion Capture Viewer."""

from __future__ import annotations

GUI_INFO = {
    "name": "C3D Motion Capture Viewer",
    "tool_name": "c3d_viewer",
    "description": "View and analyze C3D motion capture files",
    "category": "Biomechanics",
    "icon": "body",
    "pyqt6": {
        "module": "c3d_viewer.ui.pyqt6.main_window",
        "class": "C3DViewerWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "C3DViewer",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
