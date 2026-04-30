"""Keyboard shortcuts help dialog for the Unified Tools Launcher."""

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QKeySequence
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QHeaderView,
)


class KeyboardShortcutsDialog(QDialog):
    """Dialog showing available keyboard shortcuts."""

    def __init__(self, parent: Any) -> None:
        """Initialize the keyboard shortcuts dialog.

        Args:
            parent: Parent widget.
        """
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the keyboard shortcuts dialog UI."""
        self.setWindowTitle("Keyboard Shortcuts")
        self.setModal(True)
        self.resize(600, 500)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #f5f5f5;
            }
            QLabel#titleLabel {
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }
            QTableWidget {
                background-color: white;
                border: 1px solid #ccc;
                gridline-color: #e0e0e0;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QHeaderView::section {
                background-color: #e8e8e8;
                padding: 5px;
                border: none;
                border-right: 1px solid #ccc;
                border-bottom: 1px solid #ccc;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Keyboard Shortcuts")
        title.setObjectName("titleLabel")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        subtitle = QLabel("Quick reference for available keyboard shortcuts in the Unified Tools Launcher")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #666;")
        layout.addWidget(subtitle)

        # Create table
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Shortcut", "Action"])
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        table.setAlternatingRowColors(True)

        # Shortcuts data
        shortcuts = [
            ("Ctrl+F", "Open search dialog to quickly find tools"),
            ("Esc", "Clear search / Close dialog"),
            ("Tab", "Navigate to next tool"),
            ("Shift+Tab", "Navigate to previous tool"),
            ("Arrow Up/Down", "Navigate between tools"),
            ("Arrow Left/Right", "Navigate between categories"),
            ("Enter", "Launch selected tool"),
            ("F1", "Open User Manual"),
            ("Ctrl+?", "Show this keyboard shortcuts dialog"),
            ("Debug Mode Toggle", "Enable verbose logging"),
        ]

        for row, (shortcut, action) in enumerate(shortcuts):
            table.insertRow(row)

            # Shortcut cell
            shortcut_item = QTableWidgetItem(shortcut)
            shortcut_item.setFont(QFont("Courier New", 10, QFont.Weight.Bold))
            shortcut_item.setBackground(Qt.GlobalColor.lightGray)
            shortcut_item.setForeground(Qt.GlobalColor.darkBlue)
            shortcut_item.setFlags(shortcut_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, shortcut_item)

            # Action cell
            action_item = QTableWidgetItem(action)
            action_item.setFont(QFont("Segoe UI", 10))
            action_item.setFlags(action_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 1, action_item)

        table.setMaximumHeight(300)
        layout.addWidget(table)

        # Tips section
        tips_label = QLabel("Tips")
        tips_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(tips_label)

        tips_text = QLabel(
            "• Use Debug Mode to see detailed launch logs\n"
            "• Search filters are case-insensitive\n"
            "• Tool paths are validated for security\n"
            "• Failed launches show helpful error messages\n"
            "• Check the Activity Log for detailed information"
        )
        tips_text.setFont(QFont("Segoe UI", 10))
        tips_text.setStyleSheet("color: #555; line-height: 1.5;")
        tips_text.setWordWrap(True)
        layout.addWidget(tips_text)

        layout.addStretch()
