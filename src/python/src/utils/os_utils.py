"""
OS path operation utilities for consistent path handling.

This module provides reusable functions for OS path operations,
replacing direct os.path usage with Path objects where possible.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_join_path(base: Path | str, *parts: str) -> Path:
    """Safely join path parts using Path objects.

    Args:
        base: Base path
        *parts: Path parts to join

    Returns:
        Joined Path object
    """
    base_path = Path(base)
    result = base_path

    for part in parts:
        # Normalize part (remove leading slashes, handle ..)
        part = part.lstrip("/")
        if ".." in part:
            logger.warning(f"Path traversal detected in part: {part}")
        result = result / part

    return result.resolve()


def get_current_dir() -> Path:
    """Get current working directory as Path object.

    Returns:
        Current directory as Path
    """
    return Path(os.getcwd())


@contextmanager
def change_directory(path: Path | str):
    """Context manager for temporarily changing directory.

    Args:
        path: Directory to change to

    Yields:
        Path object of the new directory

    Example:
        with change_directory("/tmp"):
            # Do work in /tmp
            pass
        # Back to original directory
    """
    original_dir = Path.cwd()
    target_dir = Path(path)

    if not target_dir.exists():
        raise FileNotFoundError(f"Directory does not exist: {target_dir}")

    if not target_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {target_dir}")

    try:
        os.chdir(target_dir)
        logger.debug(f"Changed directory to: {target_dir}")
        yield target_dir
    finally:
        os.chdir(original_dir)
        logger.debug(f"Restored directory to: {original_dir}")


def path_exists(path: Path | str) -> bool:
    """Check if path exists (using Path object).

    Args:
        path: Path to check

    Returns:
        True if path exists, False otherwise
    """
    return Path(path).exists()


def ensure_dir(path: Path | str, create: bool = True) -> Path:
    """Ensure directory exists, optionally creating it.

    Args:
        path: Directory path
        create: Whether to create directory if it doesn't exist

    Returns:
        Path object of directory
    """
    dir_path = Path(path)

    if dir_path.exists():
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Path exists but is not a directory: {path}")
        return dir_path

    if create:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")
        return dir_path

    raise FileNotFoundError(f"Directory does not exist: {dir_path}")
