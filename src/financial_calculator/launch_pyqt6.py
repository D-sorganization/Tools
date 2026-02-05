#!/usr/bin/env python3
"""Standalone launcher for Financial Calculator PyQt6 GUI.

This launcher checks dependencies and starts the PyQt6 GUI application.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec


def check_dependencies() -> list[str]:
    """Check if required dependencies are available."""
    required = ["PyQt6", "numpy"]
    missing = [pkg for pkg in required if find_spec(pkg) is None]
    return missing


def main() -> int:
    """Main entry point for the Financial Calculator PyQt6 GUI."""
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}: pip install {pkg}")
        print("\nInstall the missing packages and try again.")
        return 1

    # Import and launch
    from financial_calculator.ui.pyqt6.main_window import FinancialCalculatorMainWindow
    from PyQt6.QtWidgets import QApplication

    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    app.setApplicationName("Financial Calculator")
    app.setApplicationVersion("1.0.0")

    window = FinancialCalculatorMainWindow()
    setup_themed_app(app, window, settings_app="FinancialCalculator")
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
