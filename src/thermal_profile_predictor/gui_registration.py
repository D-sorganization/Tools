"""
Thermal Profile Predictor - GUI Registration
=============================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "Thermal Profile Predictor",
    "description": "Predict temperature profiles for heated vessels",
    "category": "process_simulation",
    "version": "1.0.0",
    "entry_point": "thermal_profile_predictor.ui.pyqt6.main_window:ThermalProfilePredictorWindow",
    "icon": "thermometer",
    "keywords": [
        "thermal",
        "temperature",
        "heating",
        "vessel",
        "profile",
        "prediction",
        "ODE",
    ],
    "dependencies": {
        "required": ["PyQt6", "numpy", "scipy"],
        "optional": [],
    },
    "features": [
        "Temperature profile prediction",
        "Configurable power input function",
        "Thermal mass and heat loss parameters",
        "Time series visualization",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
