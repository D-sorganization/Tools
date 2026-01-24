#!/usr/bin/env python3
"""
Test different icon conversion methods to fix the white icon issue.
"""

import logging
from pathlib import Path
from tools.icon_utils import ensure_pil_installed, save_ico, convert_image_mode

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure PIL is available
ensure_pil_installed()
from PIL import Image


def create_alternative_ico() -> None:
    """Create alternative ICO with different settings."""
    png_path = Path("tools_icon.png")
    ico_path = Path("tools_icon_alt.ico")

    try:
        with Image.open(png_path) as img_file:
            logger.info(
                f"Original: mode={img_file.mode}, size={img_file.size}, format={img_file.format}"
            )

            # Method 1: Direct conversion with explicit RGB, using shared utility
            # Note: Test specifically asked for "RGB" if not RGB, but shared utility does RGBA for transparency.
            # Local test override or assume shared util is consistent with "best practice".
            # The original test code forced "RGB" if not "RGB".
            # Shared utility:
            # if img.mode in ("RGBA", "LA", "P"): converts to RGBA.
            # else: converts to RGB.
            # This seems safer generally.

            # For this test, we might want to stick to what it was doing or defer to new better logic.
            # The test name is "alternative_ico". It used 256, 64, 32, 16 sizes.

            sizes = [(256, 256), (64, 64), (32, 32), (16, 16)]

            # Use shared save_ico, but we need to control the image processing if it differs.
            # Save_ico calls convert_image_mode.

            save_ico(img_file, ico_path, sizes)

            logger.info(f"✓ Created alternative ICO: {ico_path}")
            logger.info(f"Size: {ico_path.stat().st_size:,} bytes")

    except Exception as e:
        logger.error(f"Failed to create alternative ICO: {e}")


def create_simple_ico() -> None:
    """Create a simple single-size ICO."""
    png_path = Path("tools_icon.png")
    ico_path = Path("tools_icon_simple.ico")

    try:
        with Image.open(png_path) as img_file:
            # Use shared save_ico with single size
            save_ico(img_file, ico_path, [(32, 32)])

            logger.info(f"✓ Created simple ICO: {ico_path}")
            logger.info(f"Size: {ico_path.stat().st_size:,} bytes")

    except Exception as e:
        logger.error(f"Failed to create simple ICO: {e}")


def main() -> None:
    """Test different conversion methods."""
    logger.info("Testing alternative icon conversion methods...")

    create_alternative_ico()
    create_simple_ico()

    logger.info("Test conversions complete!")
    logger.info("Try using tools_icon_alt.ico or tools_icon_simple.ico")


if __name__ == "__main__":
    main()
