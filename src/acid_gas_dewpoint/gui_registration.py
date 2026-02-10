"""GUI registration for Acid Gas Dewpoint Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Acid Gas Dewpoint Calculator",
    "tool_name": "acid_gas_dewpoint",
    "description": "HF, HCl, H2S dewpoint analysis for syngas applications",
    "category": "Process Simulation",
    "icon": "chemistry",
    "pyqt6": {
        "module": "acid_gas_dewpoint.python.acid_gas_dewpoint.ui.pyqt6.main_window",
        "class": "AcidGasDewpointCalculatorWidget",
        "dependencies": ["PyQt6", "matplotlib", "numpy"],
        "settings_app": "AcidGasDewpointCalculator",
        "min_size": [1000, 700],
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
