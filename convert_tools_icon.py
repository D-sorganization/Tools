#!/usr/bin/env python3
"""
Convert tools_icon.png to high-quality ICO format for Windows shortcuts.
"""

import logging
from pathlib import Path

# Use shared utility
try:
    from tools.icon_utils import convert_png_to_ico
except ImportError:
    # If package import fails (e.g. running script directly from repo root without install),
    # try to import relative to script location
    import sys

    src_path = Path(__file__).resolve().parent / "src"
    if src_path.exists():
        sys.path.append(str(src_path))
    from tools.icon_utils import convert_png_to_ico

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
            script_content = shortcut_script.read_text()
            updated_content = script_content.replace(
                "tools_icon.ico", "tools_icon_hq.ico"
            )
            shortcut_script.write_text(updated_content)
            logger.info("Updated shortcut script to use high-quality ICO")
    else:
        logger.error("Conversion failed!")


if __name__ == "__main__":
    main()
