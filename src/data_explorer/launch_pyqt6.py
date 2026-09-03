#!/usr/bin/env python3
"""Standalone PyQt6 launcher for the Data Explorer workbench."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

from shared.python.gui_launcher import make_launcher  # noqa: E402


def main() -> int:
    """Launch the Data Explorer GUI."""
    return int(make_launcher("data_explorer.gui_registration") or 0)


if __name__ == "__main__":
    sys.exit(main())
