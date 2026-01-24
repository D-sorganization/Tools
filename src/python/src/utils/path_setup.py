"""
Shared path setup utility for consistent Python path configuration.

This module provides reusable functions for setting up Python paths across
the repository, following DRY principles and ensuring consistency.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_repo_root(start_path: Path | None = None) -> Path:
    """Get the repository root directory.

    Args:
        start_path: Starting path to search from. Defaults to current file's parent.

    Returns:
        Path to repository root directory.
    """
    if start_path is None:
        start_path = Path(__file__).resolve().parent.parent.parent.parent.parent

    current = Path(start_path).resolve()
    while current != current.parent:
        # Look for common repository root indicators
        if any(
            (current / marker).exists()
            for marker in [".git", "pyproject.toml", "requirements.txt", "tools.json"]
        ):
            return current
        current = current.parent

    # Fallback: return the starting path's parent
    return Path(start_path).resolve().parent


def get_standard_paths(repo_root: Path | None = None) -> list[Path]:
    """Get standard paths to add to Python path.

    Args:
        repo_root: Repository root directory. If None, will be detected.

    Returns:
        List of paths to add to Python path.
    """
    if repo_root is None:
        repo_root = get_repo_root()

    paths_to_add = [
        repo_root,
        repo_root / "src" / "python" / "src",
        repo_root / "src" / "data_processing" / "data_processor" / "python" / "data_processor",
        repo_root / "src" / "tools",
    ]

    # Filter to only existing paths
    return [p for p in paths_to_add if p.exists()]


def setup_python_path(repo_root: Path | None = None, additional_paths: list[Path] | None = None) -> None:
    """Setup Python path for all required modules.

    Args:
        repo_root: Repository root directory. If None, will be detected.
        additional_paths: Additional paths to add beyond standard paths.
    """
    if repo_root is None:
        repo_root = get_repo_root()

    paths_to_add = get_standard_paths(repo_root)

    if additional_paths:
        paths_to_add.extend(additional_paths)

    # Add to sys.path
    for path in paths_to_add:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            logger.debug(f"Added to Python path: {path}")

    # Also set PYTHONPATH environment variable
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_paths = [str(p) for p in paths_to_add if p.exists()]

    if existing_pythonpath:
        new_pythonpath = os.pathsep.join(new_paths + [existing_pythonpath])
    else:
        new_pythonpath = os.pathsep.join(new_paths)

    os.environ["PYTHONPATH"] = new_pythonpath
    logger.debug(f"Set PYTHONPATH: {new_pythonpath}")
