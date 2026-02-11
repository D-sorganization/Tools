#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Glass Bath FEA."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)


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
