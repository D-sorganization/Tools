"""GUI registration for Flare Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Flare Calculator",
    "tool_name": "flare_calculator",
    "description": "Flare sizing and safety zone calculator",
    "category": "Process Simulation",
    "icon": "fire",
    "pyqt6": {
        "module": "flare_calculator.ui.pyqt6.main_window",
        "class": "FlareCalculatorMainWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "FlareCalculator",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
