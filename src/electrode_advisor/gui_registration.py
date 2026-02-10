"""GUI registration for Electrode Advisor."""

from __future__ import annotations

GUI_INFO = {
    "name": "Electrode Advisor",
    "tool_name": "electrode_advisor",
    "description": "AC Electrode Advancement Module for electrode system analysis",
    "category": "Process Simulation",
    "icon": "electrode",
    "pyqt6": {
        "module": "electrode_advisor.ui.pyqt6.main_window",
        "class": "ElectrodeAdvisorWidget",
        "dependencies": ["PyQt6", "numpy", "matplotlib"],
        "settings_app": "ElectrodeAdvisor",
        "min_size": [1200, 800],
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
