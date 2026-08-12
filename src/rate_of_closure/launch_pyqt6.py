#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Rate of Closure Impact Explorer."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repository root to sys.path to allow importing _bootstrap
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(__file__)

from rate_of_closure.ui.pyqt6.launcher import launch_rate_pyqt6  # noqa: E402

if __name__ == "__main__":
    sys.exit(launch_rate_pyqt6())
