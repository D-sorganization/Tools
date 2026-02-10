"""GUI registration for Multi-Parameter Analysis."""

from __future__ import annotations

GUI_INFO = {
    "name": "Multi-Parameter Analysis",
    "tool_name": "multi_param_analysis",
    "description": "Run multi-parameter sensitivity analysis with grid evaluation",
    "category": "Analysis",
    "icon": "grid",
    "pyqt6": {
        "module": "multi_param_analysis.ui.pyqt6.main_window",
        "class": "MultiParamAnalysisWindow",
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "MultiParamAnalysis",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
