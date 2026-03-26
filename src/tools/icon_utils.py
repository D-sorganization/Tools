"""Shared utilities for icon conversion and manipulation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

from src.shared.python.contracts import require

logger = logging.getLogger(__name__)


# PIL availability holder (avoids mutable globals + global keyword)
class _PILState:
    """Tracks PIL availability and provides lazy installation."""

    available: bool = False
    Image: object | None = None

    @classmethod
    def _try_import(cls) -> None:
        try:
            from PIL import Image

            cls.Image = Image
            cls.available = True
        except ImportError:
            cls.available = False

    @classmethod
    def ensure_installed(cls) -> None:
        """Ensure PIL is installed, installing it if necessary."""
        if cls.available:
            return

        from tools.dependency_utils import install_packages

        logger.info("PIL (Pillow) not found. Attempting to install...")
        if install_packages(["PIL"]):
            cls._try_import()
        else:
            logger.error("Failed to install Pillow. Icon conversion may fail.")


# Attempt import on module load
_PILState._try_import()

# Windows standard ICO sizes
ICO_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def check_pil_installed() -> bool:
    """Check if PIL/Pillow is installed."""
    return _PILState.available


def ensure_pil_installed() -> None:
    """Ensure PIL is installed, installing it if necessary."""
    _PILState.ensure_installed()


def convert_image_mode(img: Image.Image) -> Image.Image:
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
    img: Image.Image, sizes: list[tuple[int, int]]
) -> list[Image.Image]:
    """Create resized versions of an image for ICO format.

    Args:
        img: Source image to resize.
        sizes: List of (width, height) tuples for each size.

    Returns:
        List of resized images.
    """
    if not (img is not None):
        raise ValueError("img must be provided")
    require(
        isinstance(sizes, list) and len(sizes) > 0, "sizes must be a non-empty list"
    )

    from PIL import Image

    # Pillow >= 9.1.0 uses Image.Resampling; older versions use Image.LANCZOS
    lanczos = getattr(Image, "Resampling", Image).LANCZOS
    return [img.resize(size, lanczos) for size in sizes]


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
    if not (png_path is not None):
        raise ValueError("png_path must be provided")
    require(isinstance(png_path, Path), "png_path must be a Path")
    require(isinstance(ico_path, Path), "ico_path must be a Path")

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
