"""Build script for Folder Packer Pro v2.0 executable."""

import logging

# Use shared logging utility
try:
    from utils.logging_utils import init_default_logging
except ImportError:
    # Fallback
    def init_default_logging():
        logging.basicConfig(level=logging.INFO)
import subprocess

# Use shared subprocess utility
try:
    from utils.subprocess_utils import run_command
except ImportError:
    # Fallback
    import subprocess
    run_command = subprocess.run
import sys
from pathlib import Path

# Set up logging
init_default_logging()s")
logger = logging.getLogger(__name__)


def build_exe() -> int:
    """Build Windows executable using PyInstaller."""

    logger.info("=" * 60)
    logger.info("Building Folder Packer Pro v2.0 Executable")
    logger.info("=" * 60)

    # Get script directory
    script_dir = Path(__file__).parent
    main_script = script_dir / "folder_packer_pro.py"

    if not main_script.exists():
        logger.error("Error: Main script not found: %s", main_script)
        sys.exit(1)

    # PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Single executable
        "--windowed",  # No console window
        "--name=FolderPackerPro",  # Executable name
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
        run_command(cmd, cwd=script_dir, check=True)  # noqa: S603

    except subprocess.CalledProcessError as e:
        logger.exception("\nError: Build failed with exit code %s", e.returncode)
        return 1
    except FileNotFoundError:
        logger.exception("\nError: PyInstaller not found. Please install it:")
        logger.info("  pip install pyinstaller")
        return 1
    else:
        logger.info("\n%s", "=" * 60)
        logger.info("Build completed successfully!")
        logger.info("=" * 60)
        logger.info(
            "\nExecutable location: %s",
            script_dir / "dist" / "FolderPackerPro.exe",
        )

        return 0


if __name__ == "__main__":
    sys.exit(build_exe())
