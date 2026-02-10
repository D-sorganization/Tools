"""GUI registration for Financial Calculator."""

from __future__ import annotations

GUI_INFO = {
    "name": "Financial Calculator",
    "tool_name": "financial_calculator",
    "description": "Comprehensive financial modeling for plant operations",
    "category": "Process Simulation",
    "icon": "calculator",
    "pyqt6": {
        "module": "financial_calculator.ui.pyqt6.main_window",
        "class": "FinancialCalculatorMainWindow",
        "dependencies": ["PyQt6", "numpy"],
        "settings_app": "FinancialCalculator",
    },
    "engine": {
        "module": "upstream_drift_tools.process_calculators.financial_calculator",
        "class": "FinancialModelCalculator",
    },
    "web": {
        "port": 5173,
        "auto_open_browser": False,
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
