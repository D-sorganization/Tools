#!/usr/bin/env python3
"""Launch the Acid Gas Dewpoint Calculator React web application."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap  # noqa: E402

_REPO_ROOT = bootstrap(__file__)

from gui_launcher import launch_web_from_gui_info  # noqa: E402

from acid_gas_dewpoint.gui_registration import GUI_INFO  # noqa: E402

if __name__ == "__main__":
    sys.exit(launch_web_from_gui_info(GUI_INFO, __file__))
