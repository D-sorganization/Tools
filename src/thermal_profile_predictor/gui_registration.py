"""GUI registration for Thermal Profile Predictor."""

from __future__ import annotations

GUI_INFO = {
    "name": "Thermal Profile Predictor",
    "tool_name": "thermal_profile_predictor",
    "description": "Predict temperature profiles for heated vessels",
    "category": "Process Simulation",
    "icon": "thermometer",
    "pyqt6": {
        "module": "thermal_profile_predictor.ui.pyqt6.main_window",
        "class": "ThermalProfilePredictorWindow",
        "dependencies": ["PyQt6", "numpy", "scipy"],
        "settings_app": "ThermalProfilePredictor",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
