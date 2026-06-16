"""GUI registration for Steam Engine Calculator."""

from __future__ import annotations

from typing import Any

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

GUI_METADATA = {
    "name": GUI_INFO["name"],
    "description": GUI_INFO["description"],
    "category": GUI_INFO["category"].lower(),
    "entry_point": GUI_INFO["pyqt6"]["module"],
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
