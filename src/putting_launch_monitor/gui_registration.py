"""GUI registration for Putting Launch Monitor."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Putting Launch Monitor",
    "tool_name": "putting_launch_monitor",
    "description": (
        "Measure putt launch speed and direction from an overhead camera "
        "and send them to GSPro"
    ),
    "category": "Biomechanics",
    "icon": "video",
    "maturity": "beta",
    "pyqt6": {
        "module": "putting_launch_monitor.ui.pyqt6.main_window",
        "class": "PuttingMonitorWindow",
        "dependencies": ["PyQt6", "cv2", "imageio_ffmpeg", "numpy"],
        "settings_app": "PuttingLaunchMonitor",
    },
}

GUI_METADATA = {
    "name": "Putting Launch Monitor",
    "description": (
        "Measure putt launch speed and direction from an overhead camera "
        "and send them to GSPro"
    ),
    "category": "biomechanics",
    "entry_point": "putting_launch_monitor.ui.pyqt6.main_window",
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
