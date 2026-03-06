"""Entry point for running the application.

Usage:
    python -m double_pendulum_golf
"""

import sys

from PyQt6.QtWidgets import QApplication

from .gui import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # consistent cross-platform look

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
