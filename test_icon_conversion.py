#!/usr/bin/env python3
"""
Test different icon conversion methods to fix the white icon issue.
"""

import logging
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("PIL (Pillow) not found. Installing...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_alternative_ico() -> None:
    """Create alternative ICO with different settings."""
    png_path = Path("tools_icon.png")
    ico_path = Path("tools_icon_alt.ico")

    try:
        with Image.open(png_path) as img_file:
            logger.info(
                f"Original: mode={img_file.mode}, size={img_file.size}, format={img_file.format}"
            )
            img: Image.Image = img_file

            # Method 1: Direct conversion with explicit RGB
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Resize to a single standard size first
            img_256 = img.resize((256, 256), Image.Resampling.LANCZOS)

            # Save with explicit format
            img_256.save(
                ico_path, format="ICO", sizes=[(256, 256), (64, 64), (32, 32), (16, 16)]
            )

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
            # Convert to RGB and resize to 32x32
            img: Image.Image = img_file
            if img.mode != "RGB":
                img = img.convert("RGB")

            img_32 = img.resize((32, 32), Image.Resampling.LANCZOS)

            # Save as simple ICO
            img_32.save(ico_path, format="ICO")

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
