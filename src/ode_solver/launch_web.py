#!/usr/bin/env python3
"""Launch the ode_solver React web application (Vite dev server)."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

_REPO_ROOT = bootstrap(__file__)

from ode_solver.gui_registration import GUI_INFO  # noqa: E402
from shared.python.gui_launcher import launch_web_from_gui_info  # noqa: E402

if __name__ == "__main__":
    sys.exit(launch_web_from_gui_info(GUI_INFO, __file__))
