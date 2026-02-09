#!/usr/bin/env python3
"""Launch script for WGS Reactor Calculator PyQt6 application."""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap imports for development mode (before pip install -e .)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)


def main() -> int:
    """Launch the WGS Reactor Calculator PyQt6 application."""
    from wgs_reactor.ui.pyqt6.main_window import main as run_app

    run_app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
