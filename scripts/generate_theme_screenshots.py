# ruff: noqa: T201
"""Generate screenshots of the themed application for documentation.

Usage:
    QT_QPA_PLATFORM=offscreen python scripts/generate_theme_screenshots.py

This script creates screenshots of the themed application in various
states (light mode, dark mode, different color schemes) for use in
documentation and README files.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


def generate_screenshots(output_dir: Path | None = None) -> list[Path]:
    """Generate themed application screenshots.

    Args:
        output_dir: Directory to save screenshots. Defaults to docs/screenshots/.

    Returns:
        List of paths to generated screenshot files.
    """
    target_dir = output_dir or OUTPUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    screenshots: list[Path] = []

    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        from PyQt6.QtWidgets import QApplication

        _app = QApplication.instance() or QApplication(sys.argv)

        logger.info("Generating themed screenshots to %s", target_dir)

        # Placeholder: actual screenshot generation depends on the theme module
        logger.info("Screenshot generation complete: %d files", len(screenshots))
        return screenshots

    except ImportError:
        logger.error("PyQt6 is required for screenshot generation.")
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = generate_screenshots()
    print(f"Generated {len(result)} screenshots")
