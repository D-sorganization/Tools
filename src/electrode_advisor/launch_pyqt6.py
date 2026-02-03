#!/usr/bin/env python3
"""Standalone launcher for Electrode Advisor PyQt6 GUI.

This launcher checks dependencies and starts the PyQt6 GUI application.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec


def check_dependencies() -> list[str]:
    """Check if required dependencies are available."""
    required = ["PyQt6", "numpy", "matplotlib"]
    missing = [pkg for pkg in required if find_spec(pkg) is None]
    return missing


def main() -> int:
    """Main entry point for the Electrode Advisor PyQt6 GUI."""
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}: pip install {pkg}")
        print("\nInstall the missing packages and try again.")
        return 1

    # Import and launch
    from PyQt6.QtWidgets import QApplication, QMainWindow

    from electrode_advisor.ui.pyqt6.main_window import ElectrodeAdvisorWidget

    app = QApplication(sys.argv)
    app.setApplicationName("Electrode Advisor")
    app.setApplicationVersion("1.0.0")

    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Electrode Advisor - AC Electrode Advancement Module")
    window.setMinimumSize(1200, 800)

    # Create and set central widget
    advisor_widget = ElectrodeAdvisorWidget()
    window.setCentralWidget(advisor_widget)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
