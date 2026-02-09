#!/usr/bin/env python3
"""Launch script for Acid Gas Dewpoint Calculator PyQt6 GUI."""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap imports for development mode (before pip install -e .)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)


def main() -> int:
    """Launch the Acid Gas Dewpoint Calculator PyQt6 application."""
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow

        from acid_gas_dewpoint.python.acid_gas_dewpoint.ui.pyqt6.main_window import (
            AcidGasDewpointCalculatorWidget,
        )
        from shared.python.theme import setup_themed_app
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        print("Please install: pip install PyQt6 matplotlib numpy")
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("Acid Gas Dewpoint Calculator")

    window = QMainWindow()
    window.setWindowTitle("Acid Gas Dewpoint Calculator")
    window.setMinimumSize(1000, 700)

    calculator = AcidGasDewpointCalculatorWidget()
    window.setCentralWidget(calculator)
    setup_themed_app(app, window, settings_app="AcidGasDewpointCalculator")
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
