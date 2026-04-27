#!/usr/bin/env python3
"""
Convert tools_icon.png to high-quality ICO format for Windows shortcuts.
"""

import logging
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_convert_png_to_ico() -> Callable[[Path, Path], bool]:
    """Load the icon conversion helper from the installed Tools package."""
    module = import_module("tools.icon_utils")
    return cast(Callable[[Path, Path], bool], module.convert_png_to_ico)


def main() -> None:
    """Main conversion function."""
    convert_png_to_ico = load_convert_png_to_ico()
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
