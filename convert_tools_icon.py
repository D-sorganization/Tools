#!/usr/bin/env python3
"""
Convert tools_icon.png to high-quality ICO format for Windows shortcuts.
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


# Windows standard ICO sizes
ICO_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def _convert_image_mode(img: Image.Image) -> Image.Image:
    """Convert image to appropriate mode for ICO format.

    Args:
        img: PIL Image to convert.

    Returns:
        Converted image in RGBA or RGB mode.
    """
    if img.mode in ("RGBA", "LA"):
        logger.info("Image has transparency - preserving alpha channel")
        return img.convert("RGBA")
    elif img.mode == "P":
        logger.info("Converting palette image to RGBA")
        return img.convert("RGBA")
    else:
        logger.info("No transparency detected - converting to RGB")
        if img.mode != "RGB":
            return img.convert("RGB")
        return img


def _create_resized_images(
    img: Image.Image, sizes: list[tuple[int, int]]
) -> list[Image.Image]:
    """Create resized versions of an image for ICO format.

    Args:
        img: Source image to resize.
        sizes: List of (width, height) tuples for each size.

    Returns:
        List of resized images.
    """
    return [img.resize(size, Image.Resampling.LANCZOS) for size in sizes]


def convert_png_to_ico(png_path: Path, ico_path: Path) -> bool:
    """Convert PNG to high-quality ICO format."""
    try:
        if not png_path.exists():
            logger.error(f"PNG file not found: {png_path}")
            return False

        logger.info(f"Converting {png_path} to {ico_path}")

        with Image.open(png_path) as img_file:
            logger.info(f"Original image mode: {img_file.mode}, size: {img_file.size}")

            converted_img = _convert_image_mode(img_file)
            resized_images = _create_resized_images(converted_img, ICO_SIZES)

            converted_img.save(
                ico_path,
                format="ICO",
                sizes=ICO_SIZES,
                append_images=resized_images[1:],
            )

        logger.info(f"✓ Successfully converted to high-quality ICO: {ico_path}")
        logger.info(f"Original PNG size: {png_path.stat().st_size:,} bytes")
        logger.info(f"New ICO size: {ico_path.stat().st_size:,} bytes")
        return True

    except FileNotFoundError as e:
        logger.error(f"File not found during conversion: {e}")
        return False
    except PermissionError as e:
        logger.error(f"Permission denied during conversion: {e}")
        return False
    except OSError as e:
        logger.error(f"OS error during conversion: {e}")
        return False
    except ValueError as e:
        logger.error(f"Invalid image data: {e}")
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
