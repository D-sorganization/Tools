# mypy: ignore-errors
from shared.python.theme.integration import ThemedWindowMixin
import logging

logger = logging.getLogger(__name__)

#!/usr/bin/env python3
"""Minimal test to verify PyQt signals work."""

import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)
from PyQt6.QtCore import pyqtSignal


class TestWindow(ThemedWindowMixin, QMainWindow):
    run_requested = pyqtSignal()

    def __init__(self):  # type: ignore[no-untyped-def]
        super().__init__()
        self.setup_theme_support()
        self.setWindowTitle("Signal Test")
        self.setGeometry(100, 100, 300, 200)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        button = QPushButton("Click Me")
        button.clicked.connect(self.run_requested.emit)
        layout.addWidget(button)

        self.setCentralWidget(widget)

        # Connect our own handler
        self.run_requested.connect(self.on_run)

    def on_run(self):  # type: ignore[no-untyped-def]
        logger.info("[TEST] Signal received!")  # noqa: T201
        QMessageBox.information(self, "Success", "Signal was received!")


def _assert_window_contract(window: TestWindow) -> None:
    assert window.windowTitle() == "Signal Test"
    assert window.centralWidget() is not None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    _assert_window_contract(window)
    window.show()
    sys.exit(app.exec())
