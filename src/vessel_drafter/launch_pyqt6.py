#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Vessel Drafter."""

from __future__ import annotations

import sys
from pathlib import Path

_tool_dir = Path(__file__).parent
_python_dir = _tool_dir / "python"
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

from vessel_drafter.gui.vessel_drafter_window import launch  # noqa: E402

if __name__ == "__main__":
    sys.exit(launch())
