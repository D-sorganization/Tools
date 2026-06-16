"""GUI registration for Flow Rate Converter."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Flow Rate Converter",
    "tool_name": "flow_rate_converter",
    "description": "Convert between mass, molar, and volumetric flow rate units",
    "category": "Utilities",
    "icon": "exchange",
    "pyqt6": {
        "module": "flow_rate_converter.ui.pyqt6.main_window",
        "class": "FlowRateConverterWindow",
        "dependencies": ["PyQt6"],
        "settings_app": "FlowRateConverter",
    },
}

GUI_METADATA = {
    "name": GUI_INFO["name"],
    "description": GUI_INFO["description"],
    "category": GUI_INFO["category"].lower(),
    "entry_point": GUI_INFO["pyqt6"]["module"],
}


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
