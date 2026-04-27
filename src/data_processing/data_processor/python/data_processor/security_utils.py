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


ALLOWED_AST_NODES = (
    ast.BinOp
    | ast.UnaryOp
    | ast.operator
    | ast.unaryop
    | ast.cmpop
    | ast.Compare
    | ast.BoolOp
    | ast.boolop
    | ast.Constant
)


def _normalize_allowed_extensions(
    allowed_extensions: set[str] | None,
) -> set[str] | None:
    """Normalize and validate extension allow-list."""
    if allowed_extensions is None:
        return None
    normalized: set[str] = set()
    for extension in allowed_extensions:
        if not extension.startswith("."):
            raise ValueError("allowed_extensions must contain dot-prefixed extensions")
        normalized.add(extension.lower())
    return normalized


def _resolve_existing_file(file_path: str | Path) -> Path:
    """Resolve path and validate that it exists and points to a file."""
    path = Path(file_path).resolve()
    if not path.exists():
        msg = f"File does not exist: {path}"
        raise PathValidationError(msg)
    if not path.is_file():
        msg = f"Path is not a file: {path}"
        raise PathValidationError(msg)
    return path


def _validate_allowed_location(path: Path, allow_anywhere: bool) -> None:
    """Validate resolved path location against configured policy."""
    if allow_anywhere:
        return

    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    is_allowed = path.is_relative_to(cwd) or path.is_relative_to(home)
    if not is_allowed:
        msg = f"Path outside allowed directories: {path}. Allowed: {cwd} or {home}"
        raise PathValidationError(msg)


def _validate_extension(path: Path, allowed_extensions: set[str] | None) -> None:
    """Validate file suffix against a normalized allow-list."""
    if allowed_extensions is None:
        return
    suffix = path.suffix.lower()
    if suffix not in allowed_extensions:
        msg = (
            f"Unsupported file extension: {suffix}. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )
        raise PathValidationError(msg)


def _validate_name_node(node: ast.Name, allowed_names: set[str] | None) -> None:
    """Validate identifier usage against allow-list."""
    if allowed_names is not None and node.id not in allowed_names:
        msg = f"Unknown identifier: {node.id}"
        raise ExpressionValidationError(msg)


def _validate_call_node(node: ast.Call, allowed_names: set[str] | None) -> None:
    """Validate function call node and callable name."""
    if not isinstance(node.func, ast.Name):
        raise ExpressionValidationError("Complex function calls are not allowed")
    if allowed_names is not None and node.func.id not in allowed_names:
        msg = f"Unknown function: {node.func.id}"
        raise ExpressionValidationError(msg)


def _validate_expression_node(node: ast.AST, allowed_names: set[str] | None) -> None:
    """Validate an AST node against the permitted expression subset."""
    if isinstance(node, ast.Expression | ast.Load | ALLOWED_AST_NODES):
        return
    if isinstance(node, ast.Name):
        _validate_name_node(node, allowed_names)
        return
    if isinstance(node, ast.Call):
        _validate_call_node(node, allowed_names)
        return
    msg = f"Disallowed syntax: {type(node).__name__}"
    raise ExpressionValidationError(msg)


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
    if not expr.strip():
        raise ValueError("expr must not be empty")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        msg = f"Invalid syntax: {e}"
        raise ExpressionValidationError(msg) from e

    for node in ast.walk(tree):
        _validate_expression_node(node, allowed_names)


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
    if not str(file_path).strip():
        raise ValueError("file_path must not be empty")

    try:
        normalized_extensions = _normalize_allowed_extensions(allowed_extensions)
        path = _resolve_existing_file(file_path)
        _validate_allowed_location(path, allow_anywhere)
        _validate_extension(path, normalized_extensions)
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
    if max_size_bytes <= 0:
        raise ValueError("max_size_bytes must be positive")

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
    if not (file_path is not None):
        raise ValueError("file_path must be provided")
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
