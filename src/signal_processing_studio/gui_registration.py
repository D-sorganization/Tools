"""GUI registration for Signal Processing Studio."""

from __future__ import annotations

GUI_INFO = {
    "name": "Signal Processing Studio",
    "tool_name": "signal_processing_studio",
    "description": "Unified signal processing: waveform generation, analysis, filtering, curve fitting",
    "category": "Signal Processing",
    "icon": "signal",
    "pyqt6": {
        "module": "signal_processing_studio.main_window",
        "class": "SignalProcessingStudioWindow",
        "dependencies": ["PyQt6", "matplotlib", "numpy", "scipy", "sympy"],
        "settings_app": "SignalProcessingStudio",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
