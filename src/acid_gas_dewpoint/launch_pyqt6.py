#!/usr/bin/env python3
"""Launch script for Acid Gas Dewpoint Calculator PyQt6 GUI."""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directories to path for imports
TOOLS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TOOLS_ROOT / "src"))
sys.path.insert(0, str(TOOLS_ROOT / "src" / "shared" / "python"))


def main() -> int:
    """Launch the Acid Gas Dewpoint Calculator PyQt6 application."""
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow

        from acid_gas_dewpoint.python.acid_gas_dewpoint.ui.pyqt6.main_window import (
            AcidGasDewpointCalculatorWidget,
        )
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        print("Please install: pip install PyQt6 matplotlib numpy")
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("Acid Gas Dewpoint Calculator")
    app.setStyle("Fusion")

    window = QMainWindow()
    window.setWindowTitle("Acid Gas Dewpoint Calculator")
    window.setMinimumSize(1000, 700)

    calculator = AcidGasDewpointCalculatorWidget()
    window.setCentralWidget(calculator)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
