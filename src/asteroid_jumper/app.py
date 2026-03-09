"""Entry point for the Asteroid Jumper application."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from asteroid_jumper.main_window import AsteroidJumperWindow

STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #1e1e2e;
        color: #cdd6f4;
        font-family: 'Segoe UI', sans-serif;
    }
    QGroupBox {
        font-weight: bold;
        border: 1px solid #45475a;
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 10px;
        background-color: #181825;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: #89b4fa;
    }
    QLabel { color: #cdd6f4; }
    QDoubleSpinBox, QSpinBox, QComboBox {
        background-color: #313244;
        border: 1px solid #45475a;
        border-radius: 4px;
        padding: 4px;
        color: #cdd6f4;
    }
    QPushButton {
        background-color: #89b4fa;
        color: #11111b;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        font-weight: bold;
    }
    QPushButton:hover { background-color: #cba6f7; }
    QPushButton:pressed { background-color: #585b70; }
    QPushButton:disabled { background-color: #45475a; color: #585b70; }
    QProgressBar {
        background-color: #313244;
        border-radius: 3px;
        border: none;
    }
    QScrollBar:vertical {
        background-color: #181825;
        width: 10px;
    }
    QScrollBar::handle:vertical {
        background-color: #585b70;
        border-radius: 5px;
    }
    QStatusBar { background-color: #11111b; color: #a6adc8; }
    QSplitter::handle { background-color: #45475a; width: 2px; }
"""


def main() -> int:
    """Application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Asteroid Jumper")
    app.setStyleSheet(STYLESHEET)
    window = AsteroidJumperWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
