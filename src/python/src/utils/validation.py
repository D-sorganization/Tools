"""
Shared validation utilities for consistent input validation.

This module provides reusable validation functions across the repository,
following DRY principles.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def validate_path(
    path: Path | str,
    must_exist: bool = True,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    must_be_within: Path | str | None = None,
) -> tuple[bool, str | None]:
    """Validate a file or directory path.

    Args:
        path: Path to validate
        must_exist: Path must exist
        must_be_file: Path must be a file
        must_be_dir: Path must be a directory
        must_be_within: Path must be within this directory (security check)

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (path is not None):
        raise ValueError("path must be provided")
    path_obj = Path(path).resolve()

    if must_exist and not path_obj.exists():
        return False, f"Path does not exist: {path_obj}"

    if must_be_file and not path_obj.is_file():
        return False, f"Path is not a file: {path_obj}"

    if must_be_dir and not path_obj.is_dir():
        return False, f"Path is not a directory: {path_obj}"

    if must_be_within:
        base_path = Path(must_be_within).resolve()
        try:
            path_obj.relative_to(base_path)
        except ValueError:
            return (
                False,
                f"Security: Path outside allowed directory: {path_obj}",
            )

    return True, None


def validate_file_extension(
    file_path: Path | str,
    allowed_extensions: list[str],
    case_sensitive: bool = False,
) -> tuple[bool, str | None]:
    """Validate file extension.

    Args:
        file_path: Path to file
        allowed_extensions: List of allowed extensions (with or without dot)
        case_sensitive: Whether extension check is case sensitive

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
    path = Path(file_path)
    ext = path.suffix

    if not case_sensitive:
        ext = ext.lower()
        allowed_extensions = [e.lower() for e in allowed_extensions]

    # Normalize extensions (add dot if missing)
    normalized_allowed = [e if e.startswith(".") else f".{e}" for e in allowed_extensions]

    if ext not in normalized_allowed:
        return (
            False,
            f"File extension '{ext}' not allowed. Allowed: {', '.join(normalized_allowed)}",
        )

    return True, None


def validate_python_version(
    min_major: int = 3,
    min_minor: int = 10,
    min_micro: int = 0,
) -> tuple[bool, str]:
    """Validate Python version meets minimum requirements.

    Args:
        min_major: Minimum major version
        min_minor: Minimum minor version
        min_micro: Minimum micro version

    Returns:
        Tuple of (is_valid, version_string)
    """
    if not (min_major is not None):
        raise ValueError("min_major must be provided")
    import sys

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if (
        version.major < min_major
        or (version.major == min_major and version.minor < min_minor)
        or (version.major == min_major and version.minor == min_minor and version.micro < min_micro)
    ):
        required = f"{min_major}.{min_minor}.{min_micro}"
        return False, f"Python {required}+ required, found {version_str}"

    return True, version_str


def validate_not_none(value: Any, name: str = "value") -> tuple[bool, str | None]:
    """Validate that a value is not None.

    Args:
        value: Value to check
        name: Name of value for error message

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (name is not None):
        raise ValueError("name must be provided")
    if value is None:
        return False, f"{name} cannot be None"
    return True, None


def validate_not_empty(
    value: str | list | dict,
    name: str = "value",
) -> tuple[bool, str | None]:
    """Validate that a value is not empty.

    Args:
        value: Value to check (string, list, or dict)
        name: Name of value for error message

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (value is not None):
        raise ValueError("value must be provided")
    if not value:
        return False, f"{name} cannot be empty"
    return True, None


def validate_in_range(
    value: int | float,
    min_val: int | float | None = None,
    max_val: int | float | None = None,
    name: str = "value",
) -> tuple[bool, str | None]:
    """Validate that a numeric value is within range.

    Args:
        value: Value to check
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name of value for error message

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not (value is not None):
        raise ValueError("value must be provided")
    if min_val is not None and value < min_val:
        return False, f"{name} ({value}) is less than minimum ({min_val})"

    if max_val is not None and value > max_val:
        return False, f"{name} ({value}) is greater than maximum ({max_val})"

    return True, None
