"""GUI registration for Vessel Drafter."""

from __future__ import annotations

GUI_INFO = {
    "name": "Vessel Drafter",
    "tool_name": "vessel_drafter",
    "description": "Refractory vessel design with STEP, STL, BREP, and GLTF export",
    "category": "Process Simulation",
    "icon": "vessel",
    "pyqt6": {
        "module": "vessel_drafter.gui.vessel_drafter_window",
        "class": "VesselDrafterWindow",
        "launch": "launch",
        "dependencies": ["PyQt6", "numpy", "matplotlib", "build123d"],
        "settings_app": "VesselDrafter",
        "min_size": [1400, 900],
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
