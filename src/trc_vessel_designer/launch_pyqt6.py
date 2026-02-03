#!/usr/bin/env python3
"""Standalone launcher for TRC Vessel Designer PyQt6 GUI."""

from __future__ import annotations

import sys
from importlib.util import find_spec


def check_dependencies() -> list[str]:
    """Check if required dependencies are available."""
    required = ["PyQt6", "numpy"]
    missing = [pkg for pkg in required if find_spec(pkg) is None]
    return missing


def main() -> int:
    """Main entry point for the TRC Vessel Designer PyQt6 GUI."""
    missing = check_dependencies()
    if missing:
        print("Missing required packages:")
        for pkg in missing:
            print(f"  - {pkg}: pip install {pkg}")
        print("\nInstall the missing packages and try again.")
        return 1

    from PyQt6.QtWidgets import QApplication, QMainWindow

    from trc_vessel_designer.ui.pyqt6.main_window import TRCVesselDesignerWidget

    app = QApplication(sys.argv)
    app.setApplicationName("TRC Vessel Designer")
    app.setApplicationVersion("1.0.0")

    window = QMainWindow()
    window.setWindowTitle("TRC Vessel Designer - Thermal Reaction Chamber Design Tool")
    window.setMinimumSize(1200, 800)

    designer_widget = TRCVesselDesignerWidget()
    window.setCentralWidget(designer_widget)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
