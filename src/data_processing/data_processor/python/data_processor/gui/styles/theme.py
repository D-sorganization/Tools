"""Theme and styling for the PyQt6 GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QApplication

# Dark theme stylesheet
DARK_STYLESHEET = """
QWidget {
    background-color: #2b2b2b;
    color: #ffffff;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 12px;
}

QMainWindow {
    background-color: #1e1e1e;
}

QLabel {
    color: #ffffff;
    padding: 2px;
}

QPushButton {
    background-color: #0078d4;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #1084d8;
}

QPushButton:pressed {
    background-color: #006cbd;
}

QPushButton:disabled {
    background-color: #555555;
    color: #888888;
}

QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #0078d4;
}

QComboBox {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px;
    min-width: 100px;
}

QComboBox:hover {
    border-color: #0078d4;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #3c3c3c;
    color: #ffffff;
    selection-background-color: #0078d4;
}

QListWidget {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
}

QListWidget::item {
    padding: 4px;
}

QListWidget::item:selected {
    background-color: #0078d4;
}

QListWidget::item:hover {
    background-color: #404040;
}

QTableWidget {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
    gridline-color: #555555;
}

QTableWidget::item {
    padding: 4px;
}

QTableWidget::item:selected {
    background-color: #0078d4;
}

QHeaderView::section {
    background-color: #404040;
    color: #ffffff;
    padding: 6px;
    border: none;
    border-right: 1px solid #555555;
    border-bottom: 1px solid #555555;
}

QTabWidget::pane {
    border: 1px solid #555555;
    border-radius: 4px;
    background-color: #2b2b2b;
}

QTabBar::tab {
    background-color: #3c3c3c;
    color: #ffffff;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

QTabBar::tab:selected {
    background-color: #0078d4;
}

QTabBar::tab:hover:!selected {
    background-color: #404040;
}

QSpinBox, QDoubleSpinBox {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
}

QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #404040;
    border: none;
    width: 16px;
}

QScrollBar:vertical {
    background-color: #2b2b2b;
    width: 12px;
    border: none;
}

QScrollBar::handle:vertical {
    background-color: #555555;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #666666;
}

QScrollBar:horizontal {
    background-color: #2b2b2b;
    height: 12px;
    border: none;
}

QScrollBar::handle:horizontal {
    background-color: #555555;
    border-radius: 6px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #666666;
}

QStatusBar {
    background-color: #1e1e1e;
    color: #888888;
}

QMenuBar {
    background-color: #1e1e1e;
    color: #ffffff;
}

QMenuBar::item:selected {
    background-color: #0078d4;
}

QMenu {
    background-color: #2b2b2b;
    color: #ffffff;
    border: 1px solid #555555;
}

QMenu::item:selected {
    background-color: #0078d4;
}

QSplitter::handle {
    background-color: #555555;
}

QSplitter::handle:hover {
    background-color: #0078d4;
}

QGroupBox {
    border: 1px solid #555555;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
"""


def get_stylesheet() -> str:
    """Get the dark theme stylesheet."""
    return DARK_STYLESHEET


def apply_dark_theme(app: QApplication) -> None:
    """Apply dark theme to the application."""
    app.setStyleSheet(DARK_STYLESHEET)
