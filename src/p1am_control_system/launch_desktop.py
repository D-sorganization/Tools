#!/usr/bin/env python3
"""Standalone launcher for P1AM HMI Control System Desktop App."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

# Resolve import paths relative to repository structure
bootstrap(__file__)

from shared.python.gui_launcher import make_launcher  # noqa: E402

if __name__ == "__main__":
    sys.exit(make_launcher("p1am_control_system.gui_registration"))
