"""Tests for the migrated Video Analyzer launcher."""

from __future__ import annotations

from unittest.mock import patch

from video_analyzer import SwingAnalyzer, VideoProcessor
from video_analyzer.launch_video_analyzer import main


def test_video_analyzer_package_exports_core_types() -> None:
    assert SwingAnalyzer.__name__ == "SwingAnalyzer"
    assert VideoProcessor.__name__ == "VideoProcessor"


def test_video_analyzer_launcher_initializes_analyzer_without_processing() -> None:
    with patch("video_analyzer.analyzer.SwingAnalyzer", autospec=True) as analyzer_cls:
        assert main() == 0

    analyzer_cls.assert_called_once_with()
