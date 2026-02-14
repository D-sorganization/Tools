#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Thermal Profile Predictor."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap  # noqa: E402

bootstrap(__file__)

from gui_launcher import make_pyqt6_launcher  # noqa: E402

if __name__ == "__main__":
    sys.exit(make_pyqt6_launcher("thermal_profile_predictor.gui_registration"))
