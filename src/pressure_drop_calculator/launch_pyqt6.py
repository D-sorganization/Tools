#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Pressure Drop Calculator."""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap imports for development mode (before pip install -e .)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src" / "shared" / "python"))
from upstream_drift_tools.bootstrap import ensure_paths  # noqa: E402

ensure_paths(_REPO_ROOT)


def check_dependencies() -> list[str]:
    """Check for required dependencies."""
    missing = []
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        missing.append("PyQt6")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    return missing


def main() -> int:
    """Launch the Pressure Drop Calculator PyQt6 application."""
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return 1

    from PyQt6.QtWidgets import QApplication, QMainWindow

    from pressure_drop_calculator.python.pressure_drop_calculator.ui.pyqt6.main_window import (
        PressureDropCalculatorWidget,
    )
    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    app.setApplicationName("Pressure Drop Calculator")
    app.setOrganizationName("Tools")

    window = QMainWindow()
    window.setWindowTitle("Pressure Drop Calculator")
    window.setMinimumSize(1100, 700)

    widget = PressureDropCalculatorWidget(window)
    window.setCentralWidget(widget)

    setup_themed_app(app, window, settings_app="PressureDropCalculator")
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
