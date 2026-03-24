"""Utilities for managing Python package dependencies."""

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def check_dependencies(packages: list[str]) -> list[str]:
    """Check if required packages are installed.

    Args:
        packages: List of package names to check.

    Returns:
        List of missing packages.
    """
    from contracts import require

    require(isinstance(packages, list), "packages must be a list of strings")
    missing_packages = []

    for package in packages:
        try:
            # Basic import check using __import__
            # Handle special cases where import name != package name if needed
            import_name = package
            if package == "PIL":
                import_name = "PIL"  # Pillow
            elif package == "customtkinter":
                import_name = "customtkinter"

            __import__(import_name)
            logger.debug(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"✗ {package} is missing")

    return missing_packages


def install_packages(packages: list[str]) -> bool:
    """Attempt to install packages using pip.

    Args:
        packages: List of package names to install.

    Returns:
        True if all packages installed successfully, False otherwise.
    """
    from contracts import require

    require(isinstance(packages, list), "packages must be a list of strings")

    if not packages:
        return True

    logger.info(f"Attempting to install: {', '.join(packages)}")

    pip_names = {
        "PIL": "Pillow",
        "customtkinter": "customtkinter",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
    }

    try:
        success = True
        for package in packages:
            pip_name = pip_names.get(package, package)
            logger.info(f"Installing {pip_name}...")

            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", pip_name],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info(f"✓ Successfully installed {pip_name}")
            else:
                logger.error(f"✗ Failed to install {pip_name}: {result.stderr}")
                success = False

        return success

    except (subprocess.SubprocessError, OSError) as e:
        logger.error(f"Error installing packages: {e}")
        return False
