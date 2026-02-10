#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Glass Bath FEA."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Bootstrap imports for development mode (before pip install -e .)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)


def main() -> int:
    """Launch the Glass Bath FEA application."""
    try:
        from glass_bath_fea.ui.pyqt6.main_window import main as run_app

        run_app()
        return 0
    except ImportError as e:
        logger.error(f"Error importing GUI components: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
