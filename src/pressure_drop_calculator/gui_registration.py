"""GUI registration for Pressure Drop Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Pressure Drop Calculator",
    "tool_name": "pressure_drop_calculator",
    "description": "Pipe flow pressure drop analysis with multiple friction methods",
    "category": "Process Simulation",
    "icon": "pipe",
    "pyqt6": {
        "module": "pressure_drop_calculator.python.pressure_drop_calculator.ui.pyqt6.main_window",
        "class": "PressureDropCalculatorWidget",
        "dependencies": ["PyQt6", "matplotlib"],
        "settings_app": "PressureDropCalculator",
        "min_size": [1100, 700],
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
