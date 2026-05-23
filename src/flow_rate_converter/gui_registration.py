"""GUI registration for Flow Rate Converter."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Flow Rate Converter",
    "tool_name": "flow_rate_converter",
    "description": "Convert between mass, molar, and volumetric flow rate units",
    "category": "utilities",
    "icon": "exchange",
    "entry_point": "flow_rate_converter.launch_pyqt6:main",
    "pyqt6": {
        "module": "flow_rate_converter.ui.pyqt6.main_window",
        "class": "FlowRateConverterWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "FlowRateConverter",
    },
}

# Backward-compatible alias expected by tests and legacy consumers.
GUI_METADATA = GUI_INFO


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
