"""GUI registration for Video Processor."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Video Processor",
    "tool_name": "video_processor",
    "description": "Video file format conversion, frame extraction, and media analysis",
    "category": "Media Processing",
    "icon": "video",
    "web": {
        "path": "apps/web",
        "port": 3001,
        "auto_open_browser": True,
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
