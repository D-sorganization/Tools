"""Bootstrap module for launch scripts that run before package installation.

Usage in launch_pyqt6.py scripts::

    from pathlib import Path
    # One-time bootstrap: find repo root and add standard paths
    _root = Path(__file__).resolve().parents[2]  # adjust depth to reach repo root
    exec((_root / "src" / "shared" / "python" / "upstream_drift_tools" / "bootstrap.py").read_text())

Or more explicitly::

    from upstream_drift_tools.bootstrap import ensure_paths
    ensure_paths()

This module exists as a transition mechanism. Once all imports use the installed
package (via ``pip install -e .``), this module becomes unnecessary.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_paths(repo_root: Path | str | None = None) -> Path:
    """Add standard source directories to sys.path if not already present.

    This is intended for launch scripts and development use only.
    Library code should never call this function.

    Args:
        repo_root: Repository root path. Auto-detected if None.

    Returns:
        The resolved repository root path.
    """
    if repo_root is None:
        from upstream_drift_tools.utils.paths import get_repo_root

        repo_root = get_repo_root()
    else:
        repo_root = Path(repo_root).resolve()

    standard_paths = [
        repo_root / "src" / "shared" / "python",
        repo_root / "src",
        repo_root / "src" / "python" / "src",
    ]

    for path in standard_paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)

    return repo_root
