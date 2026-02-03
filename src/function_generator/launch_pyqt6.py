#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Function Generator.

This launcher provides a standalone desktop application for
generating and visualizing various waveforms.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths for imports
TOOLS_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(TOOLS_ROOT / "src"))
sys.path.insert(0, str(TOOLS_ROOT / "src" / "shared" / "python"))


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

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    return missing


def main() -> int:
    """Launch the Function Generator PyQt6 application."""
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return 1

    from PyQt6.QtWidgets import QApplication, QMainWindow

    from function_generator.python.function_generator.ui.pyqt6.main_window import (
        FunctionGeneratorWidget,
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Function Generator")
    app.setOrganizationName("Tools")

    # Create main window
    window = QMainWindow()
    window.setWindowTitle("Function Generator")
    window.setMinimumSize(1200, 700)

    # Create widget and set as central widget
    widget = FunctionGeneratorWidget(window)
    window.setCentralWidget(widget)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
