"""GUI registration for Scrubber Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Scrubber Calculator",
    "tool_name": "scrubber_calculator",
    "description": "Packed bed scrubber design with NTU/HTU mass transfer",
    "category": "Process Simulation",
    "icon": "tower",
    "pyqt6": {
        "module": "scrubber_calculator.python.scrubber_calculator.ui.pyqt6.main_window",
        "class": "ScrubberCalculatorMainWindow",
        "dependencies": ["PyQt6", "matplotlib"],
        "settings_app": "ScrubberCalculator",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
