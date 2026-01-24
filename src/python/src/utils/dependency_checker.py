"""
Shared dependency checking utility for consistent dependency management.

This module provides reusable functions for checking and installing Python
dependencies across the repository, following DRY principles.
"""

import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec

logger = logging.getLogger(__name__)


@dataclass
class DependencyStatus:
    """Represents the result of dependency checks."""

    ok: bool
    missing: list[str]
    guidance: dict[str, str]
    package_map: dict[str, str] | None = None


def check_python_version(min_major: int = 3, min_minor: int = 10) -> tuple[bool, str]:
    """Check if Python version meets minimum requirements.

    Args:
        min_major: Minimum major version required
        min_minor: Minimum minor version required

    Returns:
        Tuple of (is_valid, version_string)
    """
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major < min_major or (
        version.major == min_major and version.minor < min_minor
    ):
        return False, version_str

    return True, version_str


def has_module(name: str, spec_finder: Callable[[str], object] = find_spec) -> bool:
    """Check if a Python module can be imported.

    Args:
        name: Module name to check
        spec_finder: Function to find module spec (for testing)

    Returns:
        True if module can be imported, False otherwise
    """
    try:
        return spec_finder(name) is not None
    except Exception:
        return False


def check_dependencies(
    required: dict[str, str] | list[str],
    package_map: dict[str, str] | None = None,
    spec_finder: Callable[[str], object] = find_spec,
) -> DependencyStatus:
    """Check whether required dependencies are available.

    Args:
        required: Dictionary mapping package names to installation guidance,
                 or list of package names
        package_map: Optional mapping from import names to pip package names
        spec_finder: Function to find module spec (for testing)

    Returns:
        DependencyStatus object with check results
    """
    # Normalize input to dict format
    if isinstance(required, list):
        required_dict = {name: f"pip install {name}" for name in required}
    else:
        required_dict = required

    missing = [name for name in required_dict if not has_module(name, spec_finder)]

    return DependencyStatus(
        ok=not missing,
        missing=missing,
        guidance={name: required_dict[name] for name in missing},
        package_map=package_map,
    )


def install_package(
    package_name: str,
    package_map: dict[str, str] | None = None,
    upgrade: bool = False,
) -> bool:
    """Install a single Python package using pip.

    Args:
        package_name: Name of the package to install
        package_map: Optional mapping from import names to pip package names
        upgrade: Whether to upgrade if package already installed

    Returns:
        True if installation succeeded, False otherwise
    """
    # Map import name to pip name if needed
    pip_name = package_map.get(package_name, package_name) if package_map else package_name

    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(pip_name)

    try:
        logger.info(f"Installing {pip_name}...")
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"✓ Successfully installed {pip_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install {pip_name}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ Error installing {pip_name}: {e}")
        return False


def install_missing_packages(
    packages: list[str],
    package_map: dict[str, str] | None = None,
    upgrade: bool = False,
) -> bool:
    """Install multiple missing packages.

    Args:
        packages: List of package names to install
        package_map: Optional mapping from import names to pip package names
        upgrade: Whether to upgrade if packages already installed

    Returns:
        True if all installations succeeded, False otherwise
    """
    if not packages:
        return True

    logger.info(f"Attempting to install {len(packages)} packages...")

    success = True
    for package in packages:
        if not install_package(package, package_map, upgrade):
            success = False

    return success


def install_from_requirements(
    requirements_path: str | None = None,
    upgrade_pip: bool = True,
) -> bool:
    """Install packages from a requirements.txt file.

    Args:
        requirements_path: Path to requirements.txt file
        upgrade_pip: Whether to upgrade pip first

    Returns:
        True if installation succeeded, False otherwise
    """
    from pathlib import Path

    if requirements_path is None:
        # Try to find requirements.txt in current directory
        req_file = Path("requirements.txt")
        if not req_file.exists():
            logger.error("No requirements.txt file found")
            return False
        requirements_path = str(req_file)

    req_path = Path(requirements_path)
    if not req_path.exists():
        logger.error(f"Requirements file not found: {requirements_path}")
        return False

    try:
        # Upgrade pip first if requested
        if upgrade_pip:
            logger.info("Upgrading pip...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
            )

        # Install from requirements
        logger.info(f"Installing packages from {requirements_path}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("✓ Successfully installed all packages")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to install packages: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ Error installing packages: {e}")
        return False


def format_missing_dependencies(status: DependencyStatus) -> str:
    """Format missing dependencies into a user-friendly message.

    Args:
        status: DependencyStatus object

    Returns:
        Formatted message string
    """
    if status.ok:
        return "All dependencies are installed."

    lines = ["Missing dependencies:"]
    for name in status.missing:
        guidance = status.guidance.get(name, "")
        hint = f" ({guidance})" if guidance else ""
        lines.append(f"  - {name}{hint}")

    return "\n".join(lines)
