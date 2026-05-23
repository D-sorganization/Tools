"""GUI registration for Multi-Parameter Analysis."""

from __future__ import annotations

from typing import Any

GUI_INFO = {
    "name": "Multi-Parameter Analysis",
    "tool_name": "multi_param_analysis",
    "description": "Run multi-parameter sensitivity analysis with grid evaluation",
    "category": "analysis",
    "icon": "grid",
    "entry_point": "multi_param_analysis.launch_pyqt6:main",
    "pyqt6": {
        "module": "multi_param_analysis.ui.pyqt6.main_window",
        "class": "MultiParamAnalysisWindow",
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "MultiParamAnalysis",
    },
}

# Backward-compatible alias expected by tests and legacy consumers.
GUI_METADATA = GUI_INFO


def get_gui_info() -> dict[str, Any]:
    """Return GUI registration information."""
    return GUI_INFO
