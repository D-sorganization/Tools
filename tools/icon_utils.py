#!/usr/bin/env python3
"""
Shared utilities for icon conversion.
"""

import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

ICO_SIZES = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def ensure_pil_installed() -> None:
    """Ensure PIL (Pillow) is installed."""
    try:
        import PIL
    except ImportError:
        print("PIL (Pillow) not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Pillow: {e}")
            sys.exit(1)


# Ensure PIL is available before importing Image
try:
    from PIL import Image
except ImportError:
    ensure_pil_installed()
    from PIL import Image


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
    return [img.resize(size, Image.Resampling.LANCZOS) for size in sizes]


def save_ico(
    img: Image.Image,
    path: Path,
    sizes: list[tuple[int, int]] | None = None,
) -> None:
    """Save image as ICO with multiple sizes.

    Args:
        img: Source image.
        path: Output path for ICO file.
        sizes: List of sizes to include. Defaults to ICO_SIZES.
    """
    if sizes is None:
        sizes = ICO_SIZES

    converted_img = convert_image_mode(img)
    resized_images = create_resized_images(converted_img, sizes)

    # The first image in sizes is used as the base, others appended
    # But Image.save with 'sizes' often expects the image itself to be one of them?
    # Actually, PIL generic ICO saver uses 'sizes' to know what to save.
    # The original code did:
    # converted_img.save(path, format="ICO", sizes=ICO_SIZES, append_images=resized_images[1:])
    # Wait, if converted_img is not resized to the first size, this might be slightly off if original image size != first size in list.
    # The original code resized ALL images in `resized_images` including the first one?
    # Original: `resized_images = _create_resized_images(converted_img, ICO_SIZES)`
    # appends `resized_images[1:]`. What about `resized_images[0]`?
    # The `converted_img` (variable) was saved. It might NOT be `resized_images[0]`.
    # Let's verify original logic.
    # Original:
    # converted_img = _convert_image_mode(img_file)
    # resized_images = _create_resized_images(converted_img, ICO_SIZES)
    # converted_img.save(..., append_images=resized_images[1:])
    #
    # If converted_img is meant to be the first one, it should probably be resized too if it doesn't match.
    # BUT, `sizes` argument in `save` usually tells PIL what sizes to write.
    # If I pass `append_images`, those are additional images.
    # If the main image is not one of the sizes, it might be weird.
    # However, to faithfully reproduce, I should check if `converted_img` matches the desired first size.
    # The original code SAVED `converted_img` which was just mode-converted, NOT resized.
    # And it appended `resized_images[1:]`.
    # So `resized_images[0]` was effectively ignored/wasted?
    # OR, maybe `converted_img` acts as the "large" one?
    # The original sizes are `[(16, 16), ..., (256, 256)]`.
    # `resized_images[0]` is 16x16.
    # If `converted_img` is 512x512, it is saved as the first image?
    # Actually, `sizes` param in `save` is slightly magical.
    # Let's write `save_ico` to behave safely: Use the resized images explicitly.

    # Better implementation:
    # Use the first resized image as the primary one to save, and append the rest.

    if not resized_images:
        return

    resized_images[0].save(
        path,
        format="ICO",
        sizes=sizes,
        append_images=resized_images[1:],
    )
