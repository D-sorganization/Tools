# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.
# UPDATE: Decomposed into ui/ package.

"""
PyQt6 GUI for Two-Stage PSA System Analysis.

This GUI provides interactive visualization and analysis of PSA system
performance, including sensitivity analysis and O2 safety calculations.
"""

import sys
from PyQt6.QtWidgets import QApplication

# Expose components for backward compatibility
from .ui import (
    InputPanel,
    ResultsPanel,
    SensitivityPlotWidget,
    PFDWidget,
    PSAMainWindow,
    create_slider,
    MplCanvas,
)

__all__ = [
    "InputPanel",
    "ResultsPanel",
    "SensitivityPlotWidget",
    "PFDWidget",
    "PSAMainWindow",
    "create_slider",
    "MplCanvas",
    "main",
]

def main() -> None:
    """Main entry point for the GUI application."""
    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = PSAMainWindow()
    setup_themed_app(app, window, settings_app="PSAPackage")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
