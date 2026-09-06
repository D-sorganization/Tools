#!/usr/bin/env python3
"""Launch the P1AM DCS React frontend (Vite dev server on port 3002)."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

_REPO_ROOT = bootstrap(__file__)

from p1am_control_system.gui_registration import GUI_INFO  # noqa: E402
from shared.python.gui_launcher import launch_web_from_gui_info  # noqa: E402

if __name__ == "__main__":
    sys.exit(launch_web_from_gui_info(dict(GUI_INFO), __file__))
