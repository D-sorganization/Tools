"""Build script for Folder Fix Pro v3.0 executable."""

import logging
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def build_exe() -> int:
    """Build Windows executable using PyInstaller."""

    logger.info("=" * 60)
    logger.info("Building Folder Fix Pro v3.0 Executable")
    logger.info("=" * 60)

    # Get script directory
    script_dir = Path(__file__).parent
    main_script = script_dir / "folder_fix_pro.py"

    if not main_script.exists():
        logger.error("Error: Main script not found: %s", main_script)
        sys.exit(1)

    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Single executable
        "--windowed",  # No console window
        "--name=FolderFixPro",  # Executable name
        (
            "--icon=paper_plane_icon.ico"
            if (script_dir / "paper_plane_icon.ico").exists()
            else ""
        ),
        (
            "--add-data=paper_plane_icon.ico;."
            if (script_dir / "paper_plane_icon.ico").exists()
            else ""
        ),
        "--clean",  # Clean cache
        "--noconfirm",  # Overwrite without asking
        str(main_script),
    ]

    # Remove empty arguments
    cmd = [arg for arg in cmd if arg]

    logger.info("\nRunning PyInstaller...")
    logger.info("Command: %s\n", " ".join(cmd))

    try:
        subprocess.run(cmd, cwd=script_dir, check=True)  # noqa: S603

        logger.info("\n" + "=" * 60)
        logger.info("Build completed successfully!")
        logger.info("=" * 60)
        logger.info("\nExecutable location: %s", script_dir / "dist" / "FolderFixPro.exe")

        return 0

    except subprocess.CalledProcessError as e:
        logger.error("\nError: Build failed with exit code %s", e.returncode)
        return 1
    except FileNotFoundError:
        logger.error("\nError: PyInstaller not found. Please install it:")
        logger.error("  pip install pyinstaller")
        return 1


if __name__ == "__main__":
    sys.exit(build_exe())
