"""GUI registration for Data Processor."""

from __future__ import annotations

GUI_INFO = {
    "name": "Data Processor",
    "tool_name": "data_processor",
    "description": "Signal processing and time-series data analysis tool",
    "category": "Data Processing",
    "icon": "chart",
    "pyqt6": {
        "module": "data_processor.ui.pyqt6.main_window",
        "class": "DataProcessorMainWindow",
        "dependencies": ["PyQt6", "pandas", "numpy"],
        "settings_app": "DataProcessor",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
