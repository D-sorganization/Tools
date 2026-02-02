"""
OS path operation utilities for consistent path handling.

This module provides reusable functions for OS path operations,
replacing direct os.path usage with Path objects where possible.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)


def safe_join_path(base: Path | str, *parts: str) -> Path:
    """Safely join path parts using Path objects.

    Note:
        This function re-exports from path_helpers.safe_join_path for backward
        compatibility. The path_helpers version prevents directory traversal.

    Args:
        base: Base path
        *parts: Path parts to join

    Returns:
        Joined Path object

    Raises:
        ValueError: If path traversal detected
    """
    from utils.path_helpers import safe_join_path as _safe_join_path

    return Path(_safe_join_path(base, *parts))


def get_current_dir() -> Path:
    """Get current working directory as Path object.

    Returns:
        Current directory as Path
    """
    return Path(os.getcwd())


@contextmanager
def change_directory(path: Path | str) -> Iterator[Path]:
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

    Note:
        This function wraps file_utils.ensure_directory for backward compatibility.
        Consider using file_utils.ensure_directory for new code.

    Args:
        path: Directory path
        create: Whether to create directory if it doesn't exist

    Returns:
        Path object of directory

    Raises:
        NotADirectoryError: If path exists but is not a directory
        FileNotFoundError: If directory doesn't exist and create=False
    """
    from utils.file_utils import ensure_directory

    dir_path = Path(path)
    success = ensure_directory(dir_path, create=create)

    if success:
        return dir_path

    # Handle failure cases with appropriate exceptions
    if dir_path.exists() and not dir_path.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {path}")

    raise FileNotFoundError(f"Directory does not exist: {dir_path}")
