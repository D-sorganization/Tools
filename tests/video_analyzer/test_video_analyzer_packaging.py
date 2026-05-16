"""Packaging contract tests for the migrated video analyzer."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_config() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_video_analyzer_is_packaged_and_exposed_as_console_script() -> None:
    config = _project_config()

    scripts = config["project"]["scripts"]
    assert scripts["video-analyzer"] == "video_analyzer.launch_video_analyzer:main"

    package_includes = config["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "video_analyzer*" in package_includes


def test_video_analyzer_extra_declares_runtime_video_dependencies() -> None:
    config = _project_config()

    optional_dependencies = config["project"]["optional-dependencies"]
    video_deps = optional_dependencies["video-analyzer"]

    assert any(dep.startswith("opencv-python") for dep in video_deps)
    assert any(dep.startswith("mediapipe") for dep in video_deps)


from video_analyzer.analyzer import SwingAnalyzer


def test_video_path_valid_precondition() -> None:
    analyzer = SwingAnalyzer()
    assert analyzer._video_path_valid("test.mp4") is True
    assert analyzer._video_path_valid("") is True
    assert analyzer._video_path_valid(None) is False
