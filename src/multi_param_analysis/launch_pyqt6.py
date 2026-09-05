#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Multi-Parameter Analysis."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)
from shared.python.gui_launcher import make_launcher  # noqa: E402


def main() -> int:
    """Run standalone application launcher."""
    return int(make_launcher("multi_param_analysis.gui_registration"))


if __name__ == "__main__":
    sys.exit(main())
