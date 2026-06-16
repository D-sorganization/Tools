#!/usr/bin/env python3
"""Standalone PyQt6 launcher for ODE Solver."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

from shared.python.gui_launcher import make_pyqt6_launcher  # noqa: E402


def main() -> int:
    """Launch the ODE Solver PyQt6 GUI."""
    return int(make_pyqt6_launcher("ode_solver.gui_registration"))


if __name__ == "__main__":
    sys.exit(main())
