"""GUI registration for Rate of Closure Impact Explorer."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Rate of Closure Impact Explorer",
    "tool_name": "rate_of_closure",
    "description": (
        "Quantifies how a rotating clubhead's impact-point delivery differs "
        "from the tracked reference point (COM or geometric center): path "
        "deviation, attack-angle change, face rotation during contact, and "
        "the launch-monitor geometric-center question, with an animated 3D "
        "clubhead and rate sweeps."
    ),
    "category": "Biomechanics",
    "icon": "rotate",
    "pyqt6": {
        "module": "rate_of_closure.ui.pyqt6.main_window",
        "class": "RateOfClosureStandaloneMainWindow",
        "dependencies": ["PyQt6", "matplotlib", "numpy"],
        "settings_app": "RateOfClosure",
        "min_size": [1024, 700],
    },
    "web": {
        "port": 5193,
        "auto_open_browser": True,
    },
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
