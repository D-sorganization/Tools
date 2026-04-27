"""Smoke tests for media_processing package structure."""

from __future__ import annotations

from pathlib import Path


def test_media_processing_package_exists() -> None:
    """Verify the media_processing package directory structure exists."""
    pkg_root = Path(__file__).resolve().parents[1]
    assert (pkg_root / "video_processor").is_dir()
    assert (pkg_root / "README.md").is_file()


def test_video_processor_src_importable() -> None:
    """Verify video_processor_src has expected modules."""
    pkg_root = Path(__file__).resolve().parents[1]
    vp_src = pkg_root / "video_processor" / "python" / "video_processor_src"
    assert (vp_src / "api.py").is_file()
    assert (vp_src / "logger_utils.py").is_file()
