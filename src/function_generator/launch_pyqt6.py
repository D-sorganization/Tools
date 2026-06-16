#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Function Generator."""

from __future__ import annotations

import sys

from _bootstrap import bootstrap

bootstrap(__file__)

from shared.python.gui_launcher import make_pyqt6_launcher  # noqa: E402
from shared.python.gui_launcher.launcher import check_python_dependencies  # noqa: E402


def check_dependencies() -> list[str]:
    """Return missing Function Generator PyQt6 launcher dependencies."""
    from function_generator.gui_registration import get_gui_info

    dependencies = get_gui_info()["pyqt6"].get("dependencies", [])
    return check_python_dependencies(list(dependencies)).missing


if __name__ == "__main__":
    sys.exit(make_pyqt6_launcher("function_generator.gui_registration"))
