"""
Shared file I/O utility for consistent file operations.

This module provides reusable functions for common file operations across
the repository, following DRY principles.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def safe_read_json(file_path: Path | str, default: Any = None) -> Any:
    """Safely read a JSON file with error handling.

    Args:
        file_path: Path to JSON file
        default: Default value to return if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or default value
    """
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    path = Path(file_path)

    if not path.exists():
        logger.debug(f"JSON file not found: {path}, using default")
        return default

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        return default
    except (PermissionError, OSError) as e:
        logger.error(f"Error reading JSON file {path}: {e}")
        return default


def safe_write_json(
    file_path: Path | str,
    data: Any,
    indent: int = 2,
    create_parents: bool = True,
) -> bool:
    """Safely write data to a JSON file with error handling.

    Args:
        file_path: Path to JSON file
        data: Data to write (must be JSON serializable)
        indent: JSON indentation level
        create_parents: Whether to create parent directories if needed

    Returns:
        True if write succeeded, False otherwise
    """
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    path = Path(file_path)

    try:
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        logger.debug(f"Successfully wrote JSON to {path}")
        return True
    except TypeError as e:
        logger.error(f"Data not JSON serializable for {path}: {e}")
        return False
    except (PermissionError, OSError) as e:
        logger.error(f"Error writing JSON file {path}: {e}")
        return False


def ensure_directory(path: Path | str, create: bool = True) -> bool:
    """Ensure a directory exists.

    Args:
        path: Path to directory
        create: Whether to create directory if it doesn't exist

    Returns:
        True if directory exists (or was created), False otherwise
    """
    if not (path is not None):
        raise ValueError("path must be provided")
    dir_path = Path(path)

    if dir_path.exists():
        if dir_path.is_dir():
            return True
        logger.error(f"Path exists but is not a directory: {path}")
        return False

    if create:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {dir_path}")
            return True
        except (PermissionError, OSError) as e:
            logger.error(f"Failed to create directory {dir_path}: {e}")
            return False

    return False


def find_file_upwards(
    filename: str,
    start_path: Path | str | None = None,
    max_depth: int = 10,
) -> Path | None:
    """Find a file by searching upwards from a starting path.

    Args:
        filename: Name of file to find
        start_path: Starting directory (defaults to current directory)
        max_depth: Maximum number of parent directories to search

    Returns:
        Path to file if found, None otherwise
    """
    if not (filename is not None):
        raise ValueError("filename must be provided")
    start_path = Path.cwd() if start_path is None else Path(start_path)

    current = start_path.resolve()
    for _ in range(max_depth):
        candidate = current / filename
        if candidate.exists() and candidate.is_file():
            return candidate

        if current == current.parent:
            break
        current = current.parent

    return None


def safe_read_text(file_path: Path | str, encoding: str = "utf-8", default: str = "") -> str:
    """Safely read a text file with error handling.

    Args:
        file_path: Path to text file
        encoding: File encoding
        default: Default value to return if file doesn't exist or can't be read

    Returns:
        File contents or default value
    """
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    path = Path(file_path)

    if not path.exists():
        logger.debug(f"Text file not found: {path}, using default")
        return default

    try:
        return path.read_text(encoding=encoding)
    except (OSError, ValueError) as e:
        logger.error(f"Error reading text file {path}: {e}")
        return default


def safe_write_text(
    file_path: Path | str,
    content: str,
    encoding: str = "utf-8",
    create_parents: bool = True,
) -> bool:
    """Safely write text to a file with error handling.

    Args:
        file_path: Path to text file
        content: Content to write
        encoding: File encoding
        create_parents: Whether to create parent directories if needed

    Returns:
        True if write succeeded, False otherwise
    """
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    path = Path(file_path)

    try:
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(content, encoding=encoding)
        logger.debug(f"Successfully wrote text to {path}")
        return True
    except (PermissionError, OSError) as e:
        logger.error(f"Error writing text file {path}: {e}")
        return False
