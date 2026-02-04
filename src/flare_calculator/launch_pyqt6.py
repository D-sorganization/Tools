#!/usr/bin/env python3
"""Launch the Flare Calculator PyQt6 application."""

import sys

from flare_calculator.ui.pyqt6.main_window import FlareCalculatorMainWindow
from PyQt6.QtWidgets import QApplication


def main() -> None:
    """Entry point for the Flare Calculator PyQt6 application."""
    app = QApplication(sys.argv)
    window = FlareCalculatorMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
