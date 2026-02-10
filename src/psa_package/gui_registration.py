"""GUI registration for PSA System Analysis."""

from __future__ import annotations

GUI_INFO = {
    "name": "PSA System Analysis",
    "tool_name": "psa_package",
    "description": "Two-stage Pressure Swing Adsorption system analysis",
    "category": "Process Simulation",
    "icon": "filter",
    "pyqt6": {
        "module": "upstream_drift_tools.process_calculators.psa_package.psa_gui",
        "class": "PSAMainWindow",
        "dependencies": ["PyQt6", "numpy", "matplotlib"],
        "settings_app": "PSAPackage",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
