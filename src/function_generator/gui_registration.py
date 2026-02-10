"""GUI registration for Function Generator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Function Generator",
    "tool_name": "function_generator",
    "description": "Generate and visualize various waveforms (sine, square, triangle, etc.)",
    "category": "Signal Processing",
    "icon": "wave",
    "pyqt6": {
        "module": "function_generator.python.function_generator.ui.pyqt6.main_window",
        "class": "FunctionGeneratorWidget",
        "dependencies": ["PyQt6", "matplotlib", "numpy"],
        "settings_app": "FunctionGenerator",
        "min_size": [1200, 700],
    },
    "web": {
        "port": 5174,
        "auto_open_browser": True,
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
