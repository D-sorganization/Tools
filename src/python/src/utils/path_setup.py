"""Shared path setup utility for consistent Python path configuration.

Delegates to the canonical implementation in upstream_drift_tools.utils.paths.
This module is kept for backward compatibility.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_repo_root(start_path: Path | str | None = None) -> Path:
    """Get the repository root directory.

    Wraps the canonical implementation with backward-compatible fallback
    (returns parent directory instead of raising FileNotFoundError).
    """
    if start_path is None:
        import inspect

        frame = inspect.currentframe()
        if frame and frame.f_back:
            calling_file = frame.f_back.f_globals.get("__file__")
            if calling_file:
                start_path = Path(calling_file).parent
        if start_path is None:
            start_path = Path.cwd()

    try:
        from upstream_drift_tools.utils.paths import get_repo_root as _canonical

        return _canonical(start_path)
    except (ImportError, FileNotFoundError):
        pass

    # Minimal fallback -- kept thin to avoid duplicating the canonical impl
    current = Path(start_path).resolve()
    for _ in range(10):
        if any(
            (current / m).exists() for m in (".git", "pyproject.toml", "tools.json")
        ):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path(start_path).resolve().parent


def add_utils_to_path() -> None:
    """Add utils directory to sys.path (backward compatibility)."""
    repo_root = get_repo_root()
    utils_path = repo_root / "src" / "python" / "src"
    if utils_path.exists() and str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))


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
        repo_root
        / "src"
        / "data_processing"
        / "data_processor"
        / "python"
        / "data_processor",
        repo_root / "src" / "tools",
        repo_root / "src",
    ]

    return [p for p in paths_to_add if p.exists()]


def setup_python_path(
    repo_root: Path | None = None, additional_paths: list[Path] | None = None
) -> None:
    """Setup Python path for all required modules."""
    if repo_root is None:
        repo_root = get_repo_root()

    paths_to_add = get_standard_paths(repo_root)

    if additional_paths:
        paths_to_add.extend(additional_paths)

    for path in paths_to_add:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
            logger.debug("Added to Python path: %s", path)

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_paths = [str(p) for p in paths_to_add if p.exists()]

    if existing_pythonpath:
        new_pythonpath = os.pathsep.join(new_paths + [existing_pythonpath])
    else:
        new_pythonpath = os.pathsep.join(new_paths)

    os.environ["PYTHONPATH"] = new_pythonpath
    logger.debug("Set PYTHONPATH: %s", new_pythonpath)
