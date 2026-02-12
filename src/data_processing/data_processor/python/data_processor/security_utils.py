"""Security utilities for file operations.

This module provides security utilities for safe file handling including:
- Path validation and sanitization
- File size limit enforcement
- Directory traversal prevention
- Python expression validation (for formula builder)
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

try:
    from .constants import MAX_FILE_SIZE_BYTES
except ImportError:
    from constants import MAX_FILE_SIZE_BYTES  # type: ignore[no-redef]


class SecurityError(Exception):
    """Base exception for security-related errors."""


class PathValidationError(SecurityError):
    """Raised when file path validation fails."""


class FileSizeError(SecurityError):
    """Raised when file size exceeds limits."""


class ExpressionValidationError(SecurityError):
    """Raised when python expression validation fails."""


def validate_python_expression(
    expr: str,
    allowed_names: set[str] | None = None,
) -> None:
    """Validate a python expression for security.

    Args:
        expr: Python expression string to validate
        allowed_names: Set of allowed identifier names (variables/functions)

    Raises:
        ExpressionValidationError: If expression contains disallowed syntax or
            unknown names
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        msg = f"Invalid syntax: {e}"
        raise ExpressionValidationError(msg) from e

    for node in ast.walk(tree):
        if isinstance(node, ast.Expression | ast.Load):
            continue
        # Allow basic operators and comparisons
        if isinstance(
            node,
            (
                ast.BinOp
                | ast.UnaryOp
                | ast.operator
                | ast.unaryop
                | ast.cmpop
                | ast.Compare
                | ast.BoolOp
                | ast.boolop
            ),
        ):
            continue
        # Allow numbers, strings, constants
        if isinstance(node, ast.Constant):
            continue

        # Check names
        if isinstance(node, ast.Name):
            if allowed_names is not None and node.id not in allowed_names:
                msg = f"Unknown identifier: {node.id}"
                raise ExpressionValidationError(msg)
            continue

        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if allowed_names is not None and node.func.id not in allowed_names:
                    msg = f"Unknown function: {node.func.id}"
                    raise ExpressionValidationError(msg)
            else:
                # Call on something else than a name (e.g. (f())())
                msg = "Complex function calls are not allowed"
                raise ExpressionValidationError(msg)
            continue

        # If we get here, it's a disallowed node type (like Attribute, Import, etc.)
        msg = f"Disallowed syntax: {type(node).__name__}"
        raise ExpressionValidationError(msg)


def validate_file_path(
    file_path: str | Path,
    allowed_extensions: set[str] | None = None,
    allow_anywhere: bool = False,
) -> Path:
    """Validate and sanitize file path for security.

    Args:
        file_path: Path to validate
        allowed_extensions: Set of allowed file extensions (e.g., {'.csv', '.xlsx'})
        allow_anywhere: If False, restricts paths to current working directory
                       and user home directory

    Returns:
        Validated Path object

    Raises:
        PathValidationError: If path validation fails
    """
    try:
        # Convert to Path and resolve to absolute path
        path = Path(file_path).resolve()

        # Check if file exists
        if not path.exists():
            msg = f"File does not exist: {path}"
            raise PathValidationError(msg)

        # Check if it's a file (not directory)
        if not path.is_file():
            msg = f"Path is not a file: {path}"
            raise PathValidationError(msg)

        # Prevent directory traversal attacks
        if not allow_anywhere:
            cwd = Path.cwd().resolve()
            home = Path.home().resolve()

            # Check if path is within allowed directories
            try:
                # Python 3.9+: is_relative_to()
                if hasattr(path, "is_relative_to"):
                    is_allowed = path.is_relative_to(cwd) or path.is_relative_to(home)
                else:
                    # Fallback for Python 3.8
                    is_allowed = str(path).startswith(str(cwd)) or str(path).startswith(
                        str(home),
                    )

                if not is_allowed:
                    msg = (
                        f"Path outside allowed directories: {path}. "
                        f"Allowed: {cwd} or {home}"
                    )
                    raise PathValidationError(
                        msg,
                    )
            except ValueError as e:
                # Fallback if is_relative_to raises ValueError
                msg = f"Path validation failed: {path}"
                raise PathValidationError(msg) from e

        # Validate file extension if provided
        if allowed_extensions is not None:
            ext = path.suffix.lower()
            if ext not in allowed_extensions:
                msg = (
                    f"Unsupported file extension: {ext}. "
                    f"Allowed: {', '.join(sorted(allowed_extensions))}"
                )
                raise PathValidationError(
                    msg,
                )

        return path

    except PathValidationError:
        raise
    except (PermissionError, OSError) as e:
        msg = f"Path validation error: {e}"
        raise PathValidationError(msg) from e


def check_file_size(
    file_path: str | Path,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> None:
    """Check if file size is within acceptable limits.

    Args:
        file_path: Path to file
        max_size_bytes: Maximum allowed file size in bytes

    Raises:
        FileSizeError: If file size exceeds limit
    """
    try:
        path = Path(file_path)
        if not path.exists():
            msg = f"File does not exist: {path}"
            raise FileSizeError(msg)

        file_size = path.stat().st_size

        if file_size > max_size_bytes:
            size_gb = file_size / (1024**3)
            max_gb = max_size_bytes / (1024**3)
            msg = f"File too large: {size_gb:.2f} GB (max: {max_gb:.2f} GB)"
            raise FileSizeError(
                msg,
            )

    except FileSizeError:
        raise
    except (PermissionError, OSError) as e:
        msg = f"File size check error: {e}"
        raise FileSizeError(msg) from e


def validate_and_check_file(
    file_path: str | Path,
    allowed_extensions: set[str] | None = None,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
    allow_anywhere: bool = False,
) -> Path:
    """Validate file path and check size in one operation.

    Args:
        file_path: Path to validate
        allowed_extensions: Set of allowed file extensions
        max_size_bytes: Maximum allowed file size in bytes
        allow_anywhere: If False, restricts paths to CWD and home directory

    Returns:
        Validated Path object

    Raises:
        SecurityError: If validation or size check fails
    """
    # Validate path
    validated_path = validate_file_path(
        file_path,
        allowed_extensions=allowed_extensions,
        allow_anywhere=allow_anywhere,
    )

    # Check size
    check_file_size(validated_path, max_size_bytes=max_size_bytes)

    return validated_path


def get_safe_file_info(file_path: str | Path) -> dict[str, Any]:
    """Get safe file information after validation.

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path).resolve()

        if not path.exists():
            return {"error": "File does not exist"}

        stat = path.stat()
        size_bytes = stat.st_size
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_bytes / (1024**3)

        return {
            "name": path.name,
            "absolute_path": str(path),
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
            "size_gb": round(size_gb, 4),
            "modified_timestamp": stat.st_mtime,
            "is_file": path.is_file(),
            "extension": path.suffix.lower(),
            "within_size_limit": size_bytes <= MAX_FILE_SIZE_BYTES,
        }
    except (PermissionError, OSError) as e:
        return {"error": str(e)}
