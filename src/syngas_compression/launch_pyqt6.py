#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Syngas Compression Calculator.

This launcher provides a standalone desktop application for syngas
compression analysis using the shared engine.
"""

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
    """Launch the Syngas Compression Calculator PyQt6 application."""
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return 1

    from PyQt6.QtWidgets import QApplication, QMainWindow
    from upstream_drift_tools.process_calculators.syngas_compression_calculator import (
        create_syngas_compression_calculator,
    )

    from shared.python.theme import setup_themed_app

    app = QApplication(sys.argv)
    app.setApplicationName("Syngas Compression Calculator")
    app.setOrganizationName("Tools")

    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Syngas Compression Calculator")
    window.setMinimumSize(1200, 800)

    # Create calculator widget and set as central widget
    calculator = create_syngas_compression_calculator(window)
    window.setCentralWidget(calculator)

    setup_themed_app(app, window, settings_app="SyngasCompressionCalculator")
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
