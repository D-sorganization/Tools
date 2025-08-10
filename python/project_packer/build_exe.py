#!/usr/bin/env python3
"""Build executable for Folder Packer application."""

import importlib.util
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants for build configuration
SCRIPT_NAME: str = "folder_packer_gui.py"
EXE_NAME: str = "FolderPacker"
BUILD_DIR: str = "build"
DIST_DIR: str = "dist"
SPEC_FILE: str = "FolderPacker.spec"


def check_pyinstaller() -> bool:
    """Check if PyInstaller is available.

    Returns:
        bool: True if PyInstaller is available, False otherwise.


    """
    return importlib.util.find_spec("PyInstaller") is not None


def install_pyinstaller() -> bool:
    """Install PyInstaller if not available.

    Returns:
        bool: True if installation was successful, False otherwise.


    """
    logger.info("Installing PyInstaller...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyinstaller"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("PyInstaller installed successfully!")
    except subprocess.CalledProcessError:
        logger.exception("Failed to install PyInstaller")
        return False
    else:
        return True


def clean_build_dirs() -> None:
    """Clean build and dist directories."""
    for dir_name in [BUILD_DIR, DIST_DIR]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info("Cleaned %s directory", dir_name)


def build_executable() -> bool:
    """Build the executable using PyInstaller.

    Returns:
        bool: True if build was successful, False otherwise.


    """
    script_path = Path(SCRIPT_NAME)
    if not script_path.exists():
        logger.error("Error: %s not found!", SCRIPT_NAME)
        return False

    # Clean previous builds
    clean_build_dirs()

    # Build command - using trusted paths only
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name", EXE_NAME,
        str(script_path),
    ]

    logger.info("Building executable...")
    logger.info("Command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, cwd=Path.cwd(), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        logger.exception("Build failed")
        return False
    else:
        logger.info("Build completed successfully!")
        return True


def verify_build() -> bool:
    """Verify that the build was successful.

    Returns:
        bool: True if build verification passed, False otherwise.


    """
    exe_path = Path(DIST_DIR) / f"{EXE_NAME}.exe"
    if not exe_path.exists():
        logger.error("Error: Executable not found at %s", exe_path)
        return False

    file_size = exe_path.stat().st_size
    logger.info("Executable created: %s", exe_path)
    logger.info("File size: %.1f MB", file_size / (1024 * 1024))

    return True


def main() -> None:
    """Execute the main build process."""
    logger.info("Folder Packer - Executable Builder")
    logger.info("=" * 40)

    # Check if PyInstaller is available
    if not check_pyinstaller():
        logger.info("PyInstaller not found. Installing...")
        if not install_pyinstaller():
            logger.error("Failed to install PyInstaller. Exiting.")
            sys.exit(1)

    # Build the executable
    if build_executable():
        if verify_build():
            logger.info("\n✅ Build completed successfully!")
            logger.info("Executable location: %s", Path(DIST_DIR) / f"{EXE_NAME}.exe")
        else:
            logger.error("\n❌ Build verification failed!")
            sys.exit(1)
    else:
        logger.error("\n❌ Build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
