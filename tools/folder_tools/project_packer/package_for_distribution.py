#!/usr/bin/env python3
"""Package creation utility for Folder Packer distribution."""

import logging
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants for package creation
PACKAGE_VERSION: str = "1.1"
PACKAGE_BASE_NAME: str = "FolderPacker"
PACKAGES_DIR: str = "packages"
REQUIRED_FILES: list[str] = [
    "folder_packer_gui.py",
    "build_exe.py",
    "build.bat",
    "README.md",
    "requirements.txt",
]


def create_deployment_package() -> bool:
    """Create a deployment package."""
    current_dir = Path(__file__).parent

    # Create packages directory
    packages_dir = current_dir / PACKAGES_DIR
    packages_dir.mkdir(exist_ok=True)

    # Create package directory with timestamp
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    package_name = f"{PACKAGE_BASE_NAME}_v{PACKAGE_VERSION}_{timestamp}"
    package_dir = current_dir / PACKAGES_DIR / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    # Copy required files
    success = copy_required_files(current_dir, package_dir)
    if not success:
        return False

    # Create zip archive
    zip_path = packages_dir / f"{package_name}.zip"
    create_zip_archive(package_dir, zip_path)

    logger.info("Package created successfully: %s", zip_path)
    return True


def copy_required_files(source_dir: Path, package_dir: Path) -> bool:
    """Copy required files to package directory.

    Args:
        source_dir: Source directory containing files.
        package_dir: Package directory to copy files to.

    Returns:
        bool: True if all files were copied successfully.

    """
    for filename in REQUIRED_FILES:
        source_file = source_dir / filename
        if not source_file.exists():
            logger.warning("Warning: Required file %s not found", filename)
            continue

        dest_file = package_dir / filename
        try:
            shutil.copy2(source_file, dest_file)
            logger.info("Copied: %s", filename)
        except OSError:
            logger.exception("Error copying %s", filename)
            return False

    return True


def create_zip_archive(source_dir: Path, zip_path: Path) -> None:
    """Create a zip archive from the package directory.

    Args:
        source_dir: Directory to archive.
        zip_path: Path for the zip file.

    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                zipf.write(file_path, arcname)

    logger.info("Created archive: %s", zip_path)


def main() -> None:
    """Run package creation."""
    success = create_deployment_package()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
