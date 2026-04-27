#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Humanoid Character Builder."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

from gui_launcher import make_pyqt6_launcher  # noqa: E402

if __name__ == "__main__":
    sys.exit(make_pyqt6_launcher("humanoid_builder_gui.gui_registration"))
