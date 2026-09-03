"""GUI registration for Steam Engine Calculator."""

from __future__ import annotations

from typing import Any

GUI_INFO: dict[str, Any] = {
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
    "web": {
        "path": "web",
        "port": 5175,
        "auto_open_browser": True,
    },
}

GUI_METADATA = {
    "name": "Steam Engine Calculator",
    "description": "Steam property calculations and Rankine cycle analysis",
    "category": "thermodynamics",
    "entry_point": "steam_engine_calculator.ui.pyqt6.main_window",
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
