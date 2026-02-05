#!/usr/bin/env python3
"""Standalone launcher for Baghouse Calculator PyQt6 GUI."""

from __future__ import annotations

import sys
from importlib.util import find_spec


def check_dependencies() -> list[str]:
    """Check if required dependencies are available."""
    required = ["PyQt6", "numpy"]
    return [pkg for pkg in required if find_spec(pkg) is None]


def main() -> int:
    """Main entry point for the Baghouse Calculator PyQt6 GUI."""
    missing = check_dependencies()
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}: pip install {pkg}")
        return 1

    from baghouse_calculator.ui.pyqt6.main_window import BaghouseCalculatorMainWindow
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Baghouse Calculator")
    app.setApplicationVersion("1.0.0")

    window = BaghouseCalculatorMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
