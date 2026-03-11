#!/usr/bin/env python3
"""Standalone Tkinter GUI launcher for DWSIM Gasification Model."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

from dwsim_model.gui.main_window import launch  # noqa: E402

if __name__ == "__main__":
    sys.exit(launch())
