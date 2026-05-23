"""GUI registration for C3D Motion Capture Viewer."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "C3D Motion Capture Viewer",
    "tool_name": "c3d_viewer",
    "description": "View and analyze C3D motion capture files",
    "category": "biomechanics",
    "icon": "body",
    "entry_point": "c3d_viewer.launch_pyqt6:main",
    "pyqt6": {
        "module": "c3d_viewer.ui.pyqt6.main_window",
        "class": "C3DViewerWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "C3DViewer",
    },
}

# Backward-compatible alias expected by tests and legacy consumers.
GUI_METADATA = GUI_INFO


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
