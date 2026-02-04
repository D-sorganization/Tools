"""GUI registration for Baghouse Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Baghouse Calculator",
    "description": "Baghouse filter performance and drum sizing calculator",
    "category": "Process Simulation",
    "icon": "filter",
    "pyqt6": {
        "module": "baghouse_calculator.ui.pyqt6.main_window",
        "class": "BaghouseCalculatorMainWindow",
        "launcher": "launch_pyqt6.py",
    },
    "web": {
        "path": "web",
        "launcher": "launch_web.py",
    },
    "engine": {
        "module": "upstream_drift_tools.process_calculators.baghouse_calculator",
        "class": "BaghouseCalculator",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
