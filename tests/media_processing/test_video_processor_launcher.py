"""Tests for the Video Processor launcher registration."""

from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import patch

from media_processing.video_processor.gui_registration import GUI_INFO, get_gui_info


def test_video_processor_gui_registration_exposes_web_surface() -> None:
    assert get_gui_info() is GUI_INFO
    assert GUI_INFO["name"] == "Video Processor"
    assert GUI_INFO["tool_name"] == "video_processor"
    assert GUI_INFO["category"] == "Media Processing"
    assert GUI_INFO["web"] == {
        "path": "apps/web",
        "port": 3001,
        "auto_open_browser": True,
    }


def test_video_processor_launch_web_delegates_to_shared_launcher() -> None:
    launch_module = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "media_processing"
        / "video_processor"
        / "launch_web.py"
    )

    with (
        patch(
            "gui_launcher.launch_web_from_gui_info",
            return_value=0,
        ) as mock_launch,
        patch("sys.exit") as mock_exit,
    ):
        runpy.run_path(str(launch_module), run_name="__main__")

    mock_launch.assert_called_once_with(GUI_INFO, str(launch_module))
    mock_exit.assert_called_once_with(0)
