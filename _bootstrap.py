"""Repository-level bootstrap module.

When either ``pip install -e .`` has NOT been run **or** a launcher is
executed directly (e.g. ``python src/electrode_advisor/launch_pyqt6.py``),
the shared library directories need to be on ``sys.path``.

This module performs a **conditional** bootstrap: it silently resolves
``upstream_drift_tools.bootstrap.ensure_paths`` by first trying the
normal import (which works when the package is installed) and falling
back to computing the path relative to calling file.

Usage in launcher scripts::

    # Replace:
    #   _REPO_ROOT = Path(__file__).resolve().parents[2]
    #   sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
    #   from upstream_drift_tools.bootstrap import ensure_paths
    #   ensure_paths(_REPO_ROOT)
    #
    # With:
    from _bootstrap import bootstrap
    bootstrap(__file__)
"""

from __future__ import annotations

import sys
from pathlib import Path


def bootstrap(caller_file: str) -> Path:
    """Bootstrap import paths for a launcher or script.

    Args:
        caller_file: ``__file__`` from the calling module.

    Returns:
        The resolved repository root directory.

    Raises:
        TypeError: If caller_file is not a str.
        ValueError: If caller_file is an empty string.
    """
    if not isinstance(caller_file, str):
        raise TypeError(f"caller_file must be a str, got {type(caller_file)}")
    if not caller_file:
        raise ValueError("caller_file must not be an empty string")
    caller = Path(caller_file).resolve()
    # Walk up until we find pyproject.toml (repo root marker)
    repo_root = caller.parent
    for _ in range(10):
        if (repo_root / "pyproject.toml").exists():
            break
        repo_root = repo_root.parent
    else:
        repo_root = caller.parent

    shared_python = repo_root / "src" / "shared" / "python"
    if shared_python.exists() and str(shared_python) not in sys.path:
        sys.path.insert(0, str(shared_python))

    # Also add repo root and src/ for general imports
    for path in [str(repo_root), str(repo_root / "src")]:
        if path not in sys.path:
            sys.path.insert(0, path)

    return repo_root
