"""Smoke tests for media_processing package structure."""

from pathlib import Path

import pytest


@pytest.mark.unit
def test_media_processing_package_exists() -> None:
    """Verify the media_processing package directory exists."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "media_processing"
    assert src_root.is_dir(), "media_processing package directory must exist"


@pytest.mark.unit
def test_media_processing_has_video_processor() -> None:
    """Verify video_processor subpackage exists."""
    src_root = Path(__file__).resolve().parents[1] / "src" / "media_processing"
    assert (src_root / "video_processor").is_dir()
