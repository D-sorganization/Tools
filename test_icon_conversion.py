#!/usr/bin/env python3
"""
Test different icon conversion methods to fix the white icon issue.
"""

import logging
from pathlib import Path

# Use shared utility
try:
    from tools.icon_utils import convert_png_to_ico, ensure_pil_installed
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from tools.icon_utils import convert_png_to_ico, ensure_pil_installed

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_alternative_ico() -> None:
    """Create alternative ICO with different settings."""
    png_path = Path("tools_icon.png")
    ico_path = Path("tools_icon_alt.ico")

    # Custom sizes for alternative test
    alt_sizes = [(256, 256), (64, 64), (32, 32), (16, 16)]

    if convert_png_to_ico(png_path, ico_path, sizes=alt_sizes):
        logger.info(f"Created alternative ICO: {ico_path}")
    else:
        logger.error(f"Failed to create alternative ICO: {ico_path}")


def create_simple_ico() -> None:
    """Create a simple single-size ICO."""
    png_path = Path("tools_icon.png")
    ico_path = Path("tools_icon_simple.ico")

    # Single size test
    simple_size = [(32, 32)]

    if convert_png_to_ico(png_path, ico_path, sizes=simple_size):
        logger.info(f"Created simple ICO: {ico_path}")
    else:
        logger.error(f"Failed to create simple ICO: {ico_path}")


def main() -> None:
    """Test different conversion methods."""
    logger.info("Testing alternative icon conversion methods...")

    ensure_pil_installed()

    create_alternative_ico()
    create_simple_ico()

    logger.info("Test conversions complete!")
    logger.info("Try using tools_icon_alt.ico or tools_icon_simple.ico")


if __name__ == "__main__":
    main()
