#!/usr/bin/env python3
"""Launch script for the Data Processor PyQt6 GUI."""

from __future__ import annotations

import sys


def main() -> None:
    """Run the PyQt6 GUI application."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        print("Error: PyQt6 is required. Install with: pip install PyQt6")
        sys.exit(1)

    from data_processor.gui.main_window import DataProcessorMainWindow
    from data_processor.gui.styles.theme import apply_dark_theme

    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = DataProcessorMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
