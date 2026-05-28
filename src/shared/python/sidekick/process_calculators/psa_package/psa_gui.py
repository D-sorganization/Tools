"""PyQt6 GUI for Two-Stage PSA System Analysis.

This module is a thin backward-compatibility facade. The implementation was
decomposed into the :mod:`.ui` package; the widgets and helpers are re-exported
here so existing imports (``from ...psa_gui import PSAMainWindow``) keep working.

This GUI provides interactive visualization and analysis of PSA system
performance, including sensitivity analysis and O2 safety calculations.
"""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from .ui import (
    InputPanel,
    MplCanvas,
    PFDWidget,
    PSAMainWindow,
    ResultsPanel,
    SensitivityPlotWidget,
    create_slider,
)

__all__ = [
    "InputPanel",
    "MplCanvas",
    "PFDWidget",
    "PSAMainWindow",
    "ResultsPanel",
    "SensitivityPlotWidget",
    "create_slider",
    "main",
]

_logger = logging.getLogger(__name__)


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
