#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Inertia Calculator."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

from shared.python.gui_launcher import make_launcher  # noqa: E402


def main() -> int:
    """Launch the inertia calculator GUI."""
    return int(make_launcher("inertia_calculator.gui_registration") or 0)


if __name__ == "__main__":
    sys.exit(main())
