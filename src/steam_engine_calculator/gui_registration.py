"""GUI registration for Steam Engine Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Steam Engine Calculator",
    "tool_name": "steam_engine_calculator",
    "description": "Calculate thermodynamic properties of steam/water",
    "category": "Thermodynamics",
    "icon": "steam",
    "pyqt6": {
        "module": "steam_engine_calculator.ui.pyqt6.main_window",
        "class": "SteamEngineCalculatorWindow",
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "SteamEngineCalculator",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
