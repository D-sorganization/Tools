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


def convert_png_to_ico(png_path: Path, ico_path: Path) -> bool:
    """Convert PNG to high-quality ICO format."""
    try:
        if not png_path.exists():
            logger.error(f"PNG file not found: {png_path}")
            return False

        logger.info(f"Converting {png_path} to {ico_path}")
        
        # Open the PNG image
        with Image.open(png_path) as img:
            logger.info(f"Original image mode: {img.mode}, size: {img.size}")
            
            # Handle transparency properly
            if img.mode in ('RGBA', 'LA'):
                # Image has transparency
                logger.info("Image has transparency - preserving alpha channel")
                # Keep as RGBA for transparency
                converted_img = img.convert('RGBA')
            elif img.mode == 'P':
                # Palette mode - convert to RGBA to handle transparency
                logger.info("Converting palette image to RGBA")
                converted_img = img.convert('RGBA')
            else:
                # No transparency - convert to RGB with white background
                logger.info("No transparency detected - converting to RGB")
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                converted_img = img
            
            # Create multiple sizes for the ICO file (Windows standard sizes)
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            
            # Create a list of resized images
            resized_images = []
            for size in sizes:
                resized = converted_img.resize(size, Image.Resampling.LANCZOS)
                resized_images.append(resized)
            
            # Save as ICO with multiple sizes
            converted_img.save(
                ico_path,
                format='ICO',
                sizes=sizes,
                append_images=resized_images[1:]  # First image is the base, rest are appended
            )
            
        logger.info(f"✓ Successfully converted to high-quality ICO: {ico_path}")
        logger.info(f"Original PNG size: {png_path.stat().st_size:,} bytes")
        logger.info(f"New ICO size: {ico_path.stat().st_size:,} bytes")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert PNG to ICO: {e}")
        return False


def main():
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
            updated_content = content.replace(
                "tools_icon.ico",
                "tools_icon_hq.ico"
            )
            shortcut_script.write_text(updated_content)
            logger.info("✓ Updated shortcut script to use high-quality ICO")
    else:
        logger.error("Conversion failed!")


if __name__ == "__main__":
    main()