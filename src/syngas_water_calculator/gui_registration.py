"""GUI registration for Syngas Water Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Syngas Water Calculator",
    "tool_name": "syngas_water_calculator",
    "description": "Calculate water content and dew point in syngas systems",
    "category": "Process Simulation",
    "icon": "droplet",
    "pyqt6": {
        "module": "syngas_water_calculator.ui.pyqt6.main_window",
        "class": "SyngasWaterCalculatorWindow",
        "dependencies": ["PyQt6", "numpy", "scipy", "pandas"],
        "settings_app": "SyngasWaterCalculator",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
