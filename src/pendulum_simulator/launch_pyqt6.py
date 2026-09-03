#!/usr/bin/env python3
"""Standalone PyQt6 launcher for the Pendulum Simulator."""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import bootstrap

bootstrap(__file__)

# The application package lives one level down (src/pendulum_simulator/src).
_PACKAGE_SRC = Path(__file__).resolve().parent / "src"
if str(_PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_SRC))

from shared.python.gui_launcher import make_launcher  # noqa: E402


def main() -> int:
    """Launch the Pendulum Simulator GUI."""
    return int(make_launcher("pendulum_simulator.gui_registration") or 0)


if __name__ == "__main__":
    sys.exit(main())
