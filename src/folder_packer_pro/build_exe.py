"""Build script for Folder Packer Pro v2.0 executable."""

import logging
import subprocess
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _build_pyinstaller_command(script_dir: Path, main_script: Path) -> list[str]:
    """Build the PyInstaller command with appropriate arguments.

    Args:
        script_dir: Directory containing the script and icon.
        main_script: Path to the main Python script to build.

    Returns:
        List of command arguments for PyInstaller.
    """
    assert script_dir is not None, "script_dir must be provided"
    icon_path = script_dir / "paper_plane_icon.ico"
    has_icon = icon_path.exists()

    cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",
        "--name=FolderPackerPro",
        "--clean",
        "--noconfirm",
    ]

    if has_icon:
        cmd.extend(
            [
                "--icon=paper_plane_icon.ico",
                "--add-data=paper_plane_icon.ico;.",
            ]
        )

    cmd.append(str(main_script))
    return cmd


def _log_build_result(script_dir: Path) -> None:
    """Log successful build result with executable location."""
    logger.info("\n%s", "=" * 60)
    logger.info("Build completed successfully!")
    logger.info("=" * 60)
    logger.info(
        "\nExecutable location: %s",
        script_dir / "dist" / "FolderPackerPro.exe",
    )


def build_exe() -> int:
    """Build Windows executable using PyInstaller."""
    logger.info("=" * 60)
    logger.info("Building Folder Packer Pro v2.0 Executable")
    logger.info("=" * 60)

    script_dir = Path(__file__).parent
    main_script = script_dir / "folder_packer_pro.py"

    if not main_script.exists():
        logger.error("Error: Main script not found: %s", main_script)
        return 1

    cmd = _build_pyinstaller_command(script_dir, main_script)

    logger.info("\nRunning PyInstaller...")
    logger.info("Command: %s\n", " ".join(cmd))

    try:
        subprocess.run(cmd, cwd=script_dir, check=True)
    except subprocess.CalledProcessError as e:
        logger.exception("\nError: Build failed with exit code %s", e.returncode)
        return 1
    except FileNotFoundError:
        logger.exception("\nError: PyInstaller not found. Please install it:")
        logger.info("  pip install pyinstaller")
        return 1

    _log_build_result(script_dir)
    return 0


if __name__ == "__main__":
    sys.exit(build_exe())
