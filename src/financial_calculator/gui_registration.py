"""GUI registration for Financial Calculator.

This module provides metadata for the GUI launcher framework.
"""

from __future__ import annotations

GUI_INFO = {
    "name": "Financial Calculator",
    "description": "Comprehensive financial modeling for plant operations",
    "category": "Process Simulation",
    "icon": "calculator",
    "pyqt6": {
        "module": "financial_calculator.ui.pyqt6.main_window",
        "class": "FinancialCalculatorMainWindow",
        "launcher": "launch_pyqt6.py",
    },
    "web": {
        "path": "web",
        "launcher": "launch_web.py",
    },
    "engine": {
        "module": "upstream_drift_tools.process_calculators.financial_calculator",
        "class": "FinancialModelCalculator",
    },
}


def get_gui_info() -> dict:
    """Return GUI registration information."""
    return GUI_INFO
