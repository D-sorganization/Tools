"""
Path helper utilities for common path operations.

This module provides convenience functions for common path operations
that are frequently duplicated across the codebase.
"""

from pathlib import Path

from .path_setup import add_utils_to_path, get_repo_root


def get_file_dir(file_path: str | Path) -> Path:
    """Get the directory containing a file.

    Args:
        file_path: Path to file (typically __file__)

    Returns:
        Directory containing the file
    """
    return Path(file_path).resolve().parent


def get_project_root_from_file(file_path: str | Path) -> Path:
    """Get project root starting from a file path.

    This is a convenience wrapper around get_repo_root().

    Args:
        file_path: Path to file (typically __file__)

    Returns:
        Repository root directory
    """
    return get_repo_root(Path(file_path).parent)


def ensure_utils_in_path() -> None:
    """Ensure utils directory is in sys.path.

    This is an alias for add_utils_to_path() for convenience.
    """
    add_utils_to_path()


def get_relative_path(from_path: Path | str, to_path: Path | str) -> Path:
    """Get relative path from one location to another.

    Args:
        from_path: Starting path
        to_path: Target path

    Returns:
        Relative path from from_path to to_path
    """
    from_p = Path(from_path).resolve()
    to_p = Path(to_path).resolve()

    try:
        return to_p.relative_to(from_p)
    except ValueError:
        # Paths are not relative, return absolute
        return to_p


def find_nearest_file(filename: str, start_path: Path | str | None = None) -> Path | None:
    """Find the nearest file with given name by searching upwards.

    Args:
        filename: Name of file to find
        start_path: Starting directory (defaults to current directory)

    Returns:
        Path to file if found, None otherwise
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)

    current = start_path.resolve()
    max_depth = 20
    depth = 0

    while current != current.parent and depth < max_depth:
        candidate = current / filename
        if candidate.exists() and candidate.is_file():
            return candidate
        current = current.parent
        depth += 1

    return None


def find_nearest_dir(dirname: str, start_path: Path | str | None = None) -> Path | None:
    """Find the nearest directory with given name by searching upwards.

    Args:
        dirname: Name of directory to find
        start_path: Starting directory (defaults to current directory)

    Returns:
        Path to directory if found, None otherwise
    """
    if start_path is None:
        start_path = Path.cwd()
    else:
        start_path = Path(start_path)

    current = start_path.resolve()
    max_depth = 20
    depth = 0

    while current != current.parent and depth < max_depth:
        candidate = current / dirname
        if candidate.exists() and candidate.is_dir():
            return candidate
        current = current.parent
        depth += 1

    return None


def normalize_path(path: Path | str) -> Path:
    """Normalize a path (resolve and make absolute).

    Args:
        path: Path to normalize

    Returns:
        Normalized Path object
    """
    return Path(path).resolve()


def safe_join_path(base: Path | str, *parts: str) -> Path:
    """Safely join path parts, preventing directory traversal.

    Args:
        base: Base path
        *parts: Path parts to join

    Returns:
        Joined path

    Raises:
        ValueError: If path traversal detected
    """
    base_path = Path(base).resolve()
    result = base_path

    for part in parts:
        # Check for path traversal attempts
        if ".." in part or part.startswith("/"):
            raise ValueError(f"Unsafe path part detected: {part}")
        result = result / part

    # Ensure result is still within base
    try:
        result.resolve().relative_to(base_path)
    except ValueError:
        raise ValueError(f"Path traversal detected: {result}")

    return result.resolve()
