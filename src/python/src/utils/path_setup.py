"""Shared path setup utility for consistent Python path configuration.

.. deprecated::
    This module is retained for backward compatibility only.  New code should
    use ``pip install -e .`` and import packages directly.  The ``sys.path``
    mutation helpers (:func:`add_utils_to_path`, :func:`setup_python_path`)
    now only set the ``PYTHONPATH`` environment variable for child processes
    and log deprecation warnings instead of mutating ``sys.path`` at runtime.
    See issue #677 / #682.

Delegates to the canonical implementation in upstream_drift_tools.utils.paths.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

_DEPRECATION_MSG = (
    "{func}() is deprecated. Install the package with `pip install -e .` "
    "and use standard imports instead. See issue #682."
)


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

        return Path(_canonical(start_path))
    except (ImportError, FileNotFoundError):
        pass

    # Minimal fallback -- kept thin to avoid duplicating the canonical impl
    current = Path(start_path).resolve()
    for _ in range(10):
        if any((current / m).exists() for m in (".git", "pyproject.toml", "tools.json")):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path(start_path).resolve().parent


def add_utils_to_path() -> None:
    """Add utils directory to sys.path (backward compatibility).

    .. deprecated::
        Use ``pip install -e .`` instead.  This function now only
        updates ``PYTHONPATH`` for child processes.
    """
    warnings.warn(
        _DEPRECATION_MSG.format(func="add_utils_to_path"),
        DeprecationWarning,
        stacklevel=2,
    )
    repo_root = get_repo_root()
    utils_path = repo_root / "src" / "python" / "src"
    if utils_path.exists():
        _update_pythonpath_env([utils_path])


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
        repo_root / "src",
    ]

    return [p for p in paths_to_add if p.exists()]


def _update_pythonpath_env(paths: list[Path]) -> None:
    """Update the PYTHONPATH environment variable for child processes.

    Does NOT mutate ``sys.path`` -- callers that need runtime import
    resolution should install the package instead.
    """
    existing = os.environ.get("PYTHONPATH", "")
    new_parts = [str(p) for p in paths if p.exists()]

    if existing:
        merged = os.pathsep.join(new_parts + [existing])
    else:
        merged = os.pathsep.join(new_parts)

    os.environ["PYTHONPATH"] = merged
    logger.debug("Set PYTHONPATH: %s", merged)


def setup_python_path(
    repo_root: Path | None = None, additional_paths: list[Path] | None = None
) -> None:
    """Setup Python path for all required modules.

    .. deprecated::
        Use ``pip install -e .`` instead.  This function now only
        updates ``PYTHONPATH`` for child processes and no longer mutates
        ``sys.path``.
    """
    warnings.warn(
        _DEPRECATION_MSG.format(func="setup_python_path"),
        DeprecationWarning,
        stacklevel=2,
    )
    if repo_root is None:
        repo_root = get_repo_root()

    paths_to_add = get_standard_paths(repo_root)

    if additional_paths:
        paths_to_add.extend(additional_paths)

    _update_pythonpath_env(paths_to_add)
