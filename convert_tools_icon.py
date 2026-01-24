#!/usr/bin/env python3
"""
Convert tools_icon.png to high-quality ICO format for Windows shortcuts.
"""

import logging
from pathlib import Path

from tools.icon_utils import ICO_SIZES, ensure_pil_installed, save_ico

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Ensure PIL is available
ensure_pil_installed()
from PIL import Image


def convert_png_to_ico(png_path: Path, ico_path: Path) -> bool:
    """Convert PNG to high-quality ICO format."""
    try:
        if not png_path.exists():
            logger.error(f"PNG file not found: {png_path}")
            return False

        logger.info(f"Converting {png_path} to {ico_path}")

        with Image.open(png_path) as img_file:
            logger.info(f"Original image mode: {img_file.mode}, size: {img_file.size}")
            save_ico(img_file, ico_path, ICO_SIZES)

        logger.info(f"✓ Successfully converted to high-quality ICO: {ico_path}")
        logger.info(f"Original PNG size: {png_path.stat().st_size:,} bytes")
        logger.info(f"New ICO size: {ico_path.stat().st_size:,} bytes")
        return True

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        return False


def main() -> None:
    """Main conversion function."""
    png_path = Path("tools_icon.png")
    ico_path = Path("tools_icon_hq.ico")

    if convert_png_to_ico(png_path, ico_path):
        logger.info("Conversion completed successfully!")
        logger.info(f"You can now use {ico_path} for better quality icons")

        # Update the shortcut script to use the new high-quality ICO
        shortcut_script = Path("create_launcher_shortcut.ps1")
        if shortcut_script.exists():
            content = shortcut_script.read_text()
            updated_content = content.replace("tools_icon.ico", "tools_icon_hq.ico")
            shortcut_script.write_text(updated_content)
            logger.info("✓ Updated shortcut script to use high-quality ICO")
    else:
        logger.error("Conversion failed!")


if __name__ == "__main__":
    main()
