#!/usr/bin/env python3
"""Standalone PyQt6 launcher for C3D Motion Capture Viewer."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

from gui_launcher import make_launcher  # noqa: E402


def main() -> int:
    """Launch the C3D viewer GUI."""
    return int(make_launcher("c3d_viewer.gui_registration") or 0)


if __name__ == "__main__":
    sys.exit(main())
