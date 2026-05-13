"""GUI registration for Video Analyzer."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Video Analyzer",
    "tool_name": "video_analyzer",
    "description": "Video-based motion analysis with pose tracking",
    "category": "Motion Capture",
    "icon": "video",
    "pyqt6": {
        "module": "video_analyzer.launch_video_analyzer",
        "class": "main",
        "dependencies": ["PyQt6", "cv2", "mediapipe"],
        "settings_app": "VideoAnalyzer",
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
