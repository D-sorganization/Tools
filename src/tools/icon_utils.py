"""Shared utilities for icon conversion and manipulation."""

import logging
from pathlib import Path

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)

# Windows standard ICO sizes
ICO_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def check_pil_installed() -> bool:
    """Check if PIL/Pillow is installed."""
    return HAS_PIL


def ensure_pil_installed() -> None:
    """Ensure PIL is installed, installing it if necessary."""
    global Image, HAS_PIL

    if HAS_PIL:
        return

    from tools.dependency_utils import install_packages

    logger.info("PIL (Pillow) not found. Attempting to install...")
    if install_packages(["PIL"]):
        from PIL import Image

        HAS_PIL = True
    else:
        logger.error("Failed to install Pillow. Icon conversion may fail.")


def convert_image_mode(img: "Image.Image") -> "Image.Image":
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


def create_resized_images(
    img: "Image.Image", sizes: list[tuple[int, int]]
) -> list["Image.Image"]:
    """Create resized versions of an image for ICO format.

    Args:
        img: Source image to resize.
        sizes: List of (width, height) tuples for each size.

    Returns:
        List of resized images.
    """
    return [img.resize(size, Image.Resampling.LANCZOS) for size in sizes]


def convert_png_to_ico(
    png_path: Path, ico_path: Path, sizes: list[tuple[int, int]] | None = None
) -> bool:
    """Convert PNG to high-quality ICO format.

    Args:
        png_path: Path to source PNG file.
        ico_path: Path to destination ICO file.
        sizes: Optional list of sizes to include. Defaults to standard Windows sizes.

    Returns:
        True if successful, False otherwise.
    """
    ensure_pil_installed()
    from PIL import Image  # Ensure import is available

    if sizes is None:
        sizes = ICO_SIZES

    try:
        if not png_path.exists():
            logger.error(f"PNG file not found: {png_path}")
            return False

        logger.info(f"Converting {png_path} to {ico_path}")

        with Image.open(png_path) as img_file:
            logger.info(f"Original image mode: {img_file.mode}, size: {img_file.size}")

            converted_img = convert_image_mode(img_file)
            resized_images = create_resized_images(converted_img, sizes)

            converted_img.save(
                ico_path,
                format="ICO",
                sizes=sizes,
                append_images=resized_images[1:],
            )

        logger.info(f"Successfully converted to high-quality ICO: {ico_path}")
        if png_path.exists():
            logger.info(f"Original PNG size: {png_path.stat().st_size:,} bytes")
        if ico_path.exists():
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
