#!/usr/bin/env python3
"""Standalone PyQt6 launcher for Syngas Compression Calculator.

This launcher provides a standalone desktop application for syngas
compression analysis using the shared engine.
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

    return missing


def main() -> int:
    """Launch the Syngas Compression Calculator PyQt6 application."""
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return 1

    from PyQt6.QtWidgets import QApplication, QMainWindow

    from shared.python.upstream_drift_tools.process_calculators.syngas_compression_calculator import (
        create_syngas_compression_calculator,
    )

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

    # Apply dark theme styling
    window.setStyleSheet(
        """
        QMainWindow {
            background-color: #1e1e2e;
        }
        QWidget {
            background-color: #1e1e2e;
            color: #cdd6f4;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QGroupBox {
            border: 1px solid #45475a;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 8px;
            background-color: #313244;
        }
        QGroupBox::title {
            color: #cba6f7;
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #89b4fa;
            color: #1e1e2e;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #b4befe;
        }
        QPushButton:pressed {
            background-color: #7287fd;
        }
        QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
            background-color: #45475a;
            border: 1px solid #585b70;
            border-radius: 4px;
            padding: 4px 8px;
            color: #cdd6f4;
        }
        QTabWidget::pane {
            border: 1px solid #45475a;
            border-radius: 4px;
            background-color: #313244;
        }
        QTabBar::tab {
            background-color: #45475a;
            color: #cdd6f4;
            padding: 8px 16px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: #89b4fa;
            color: #1e1e2e;
        }
        QTableWidget {
            background-color: #313244;
            border: 1px solid #45475a;
            gridline-color: #45475a;
        }
        QTableWidget::item {
            padding: 4px;
        }
        QHeaderView::section {
            background-color: #45475a;
            color: #cdd6f4;
            padding: 4px;
            border: none;
        }
        QScrollBar:vertical {
            background-color: #313244;
            width: 12px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #585b70;
            border-radius: 6px;
            min-height: 20px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #6c7086;
        }
    """
    )

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
