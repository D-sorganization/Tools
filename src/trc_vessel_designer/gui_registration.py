"""GUI registration for TRC Vessel Designer."""

from __future__ import annotations

GUI_INFO = {
    "name": "TRC Vessel Designer",
    "tool_name": "trc_vessel_designer",
    "description": "Thermal Reaction Chamber vessel design tool",
    "category": "Process Simulation",
    "icon": "vessel",
    "pyqt6": {
        "module": "trc_vessel_designer.ui.pyqt6.main_window",
        "class": "TRCVesselDesignerWidget",
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "TRCVesselDesigner",
        "min_size": [1200, 800],
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
