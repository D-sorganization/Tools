#!/usr/bin/env python3
"""Standalone PyQt6 launcher for C3D Motion Capture Viewer."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

from gui_launcher import make_launcher  # noqa: E402


def main() -> None:
    """Entry point for the launcher (used by tests and package scripts)."""
    import sys

    launcher_module = sys.modules.get("gui_launcher")
    if launcher_module is not None:
        from gui_launcher import make_pyqt6_launcher

        sys.exit(make_pyqt6_launcher("c3d_viewer.gui_registration"))


if __name__ == "__main__":
    sys.exit(make_launcher("c3d_viewer.gui_registration"))
