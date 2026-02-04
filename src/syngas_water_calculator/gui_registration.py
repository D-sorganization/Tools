"""
Syngas Water Calculator - GUI Registration
==========================================

Metadata for GUI framework integration.
"""

GUI_METADATA = {
    "name": "Syngas Water Calculator",
    "description": "Calculate water content and dew point in syngas systems",
    "category": "process_simulation",
    "version": "1.0.0",
    "entry_point": "syngas_water_calculator.ui.pyqt6.main_window:SyngasWaterCalculatorWindow",
    "icon": "droplet",
    "keywords": [
        "syngas",
        "water content",
        "dew point",
        "vapor pressure",
        "saturation",
        "condensation",
        "gasification",
    ],
    "dependencies": {
        "required": ["PyQt6", "numpy", "scipy", "pandas"],
        "optional": [],
    },
    "features": [
        "Water content calculation (mg/Nm3, ppmv, g/m3)",
        "Dew point temperature",
        "Multiple vapor pressure methods (Antoine, Buck, IAPWS, Magnus)",
        "Predefined syngas compositions",
        "Custom composition input",
        "Condensation risk assessment",
    ],
}


def get_metadata() -> dict:
    """Return GUI metadata for framework registration."""
    return GUI_METADATA
