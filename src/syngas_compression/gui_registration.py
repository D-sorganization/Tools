"""GUI registration for Syngas Compression Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Syngas Compression Calculator",
    "tool_name": "syngas_compression",
    "description": "Multi-stage compression analysis with water dropout calculations",
    "category": "Process Simulation",
    "icon": "compress",
    "pyqt6": {
        "module": "upstream_drift_tools.process_calculators.syngas_compression_calculator",
        "class": "SyngasCompressionCalculatorWidget",
        "dependencies": ["PyQt6", "matplotlib"],
        "settings_app": "SyngasCompressionCalculator",
        "min_size": [1200, 800],
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
