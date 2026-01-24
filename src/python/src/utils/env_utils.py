"""
Environment variable and .env file utilities.

This module provides reusable functions for handling environment variables
and .env files across the repository, following DRY principles.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def find_env_file(
    filename: str = ".env",
    start_path: Path | str | None = None,
    search_locations: list[Path | str] | None = None,
) -> Path | None:
    """Find .env file in common locations.

    Args:
        filename: Name of env file to find (default: .env)
        start_path: Starting path for search
        search_locations: Additional locations to search

    Returns:
        Path to .env file if found, None otherwise
    """
    locations: list[Path] = []

    # Add custom search locations
    if search_locations:
        locations.extend([Path(loc) for loc in search_locations])

    # Add standard locations
    if start_path:
        start = Path(start_path)
        locations.append(start / filename)
        locations.append(start.parent / filename)
    else:
        locations.append(Path.cwd() / filename)
        locations.append(Path(__file__).parent.parent.parent.parent.parent / filename)

    # Add user home directory
    locations.append(Path.home() / ".pdf_renamer" / filename)

    # Search for file
    for loc in locations:
        if loc.exists() and loc.is_file():
            logger.debug(f"Found env file: {loc}")
            return loc

    return None


def load_env_file(
    env_path: Path | str | None = None,
    filename: str = ".env",
    search_locations: list[Path | str] | None = None,
) -> bool:
    """Load environment variables from .env file.

    Args:
        env_path: Explicit path to .env file
        filename: Name of env file if searching
        search_locations: Additional locations to search

    Returns:
        True if file was loaded, False otherwise
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")
        return False

    if env_path:
        env_file = Path(env_path)
    else:
        found_file = find_env_file(filename, search_locations=search_locations)
        env_file = found_file if found_file else None

    if env_file and env_file.exists():
        load_dotenv(env_file)
        logger.debug(f"Loaded env file: {env_file}")
        return True

    return False


def get_env_var(
    key: str,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    """Get environment variable with optional default and validation.

    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether variable is required (raises if missing)

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required variable is missing
    """
    value = os.environ.get(key, default)

    if required and value is None:
        raise ValueError(f"Required environment variable {key} is not set")

    return value


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Boolean value
    """
    value = os.environ.get(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Integer value

    Raises:
        ValueError: If value cannot be converted to int
    """
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"Environment variable {key} must be an integer: {e}") from e
