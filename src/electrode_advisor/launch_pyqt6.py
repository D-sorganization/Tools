#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Electrode Advisor."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap  # noqa: E402

bootstrap(__file__)

from gui_launcher import make_pyqt6_launcher  # noqa: E402


def check_dependencies() -> list[str]:
    """Return a list of missing dependency names, or empty list if all present."""
    missing: list[str] = []
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        missing.append("PyQt6")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    return missing


if __name__ == "__main__":
    sys.exit(make_pyqt6_launcher("electrode_advisor.gui_registration"))
