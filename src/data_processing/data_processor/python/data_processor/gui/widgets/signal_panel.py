"""Signal selection panel widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass


class SignalPanel(QWidget):
    """Panel for selecting signals/columns from data."""

    # Signals
    selection_changed = pyqtSignal(list)  # List of selected signal names

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the signal panel."""
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        self._add_title(layout)
        self._add_buttons(layout)
        self._add_signal_list(layout)

    def _add_title(self, layout: QVBoxLayout) -> None:
        """Add title label."""
        title = QLabel("Signal Selection")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

    def _add_buttons(self, layout: QVBoxLayout) -> None:
        """Add action buttons."""
        button_layout = QHBoxLayout()

        self.select_all_button = QPushButton("Select All")
        self.clear_selection_button = QPushButton("Clear Selection")

        button_layout.addWidget(self.select_all_button)
        button_layout.addWidget(self.clear_selection_button)

        layout.addLayout(button_layout)

    def _add_signal_list(self, layout: QVBoxLayout) -> None:
        """Add signal list widget."""
        self.signal_list = QListWidget()
        self.signal_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.signal_list.setMinimumHeight(150)
        layout.addWidget(self.signal_list)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.select_all_button.clicked.connect(self._on_select_all)
        self.clear_selection_button.clicked.connect(self._on_clear_selection)
        self.signal_list.itemSelectionChanged.connect(self._on_selection_changed)

    def _on_select_all(self) -> None:
        """Select all items in the list."""
        self.signal_list.selectAll()

    def _on_clear_selection(self) -> None:
        """Clear all selections."""
        self.signal_list.clearSelection()

    def _on_selection_changed(self) -> None:
        """Handle selection change."""
        selected = self.get_selected_signals()
        self.selection_changed.emit(selected)

    def set_signals(self, signals: list[str]) -> None:
        """Set the list of available signals."""
        self.signal_list.clear()
        for signal in signals:
            item = QListWidgetItem(signal)
            self.signal_list.addItem(item)

    def get_selected_signals(self) -> list[str]:
        """Get list of selected signal names."""
        selected_items = self.signal_list.selectedItems()
        return [item.text() for item in selected_items]

    def get_all_signals(self) -> list[str]:
        """Get all signal names."""
        return [
            self.signal_list.item(i).text() for i in range(self.signal_list.count())
        ]

    def clear(self) -> None:
        """Clear all signals."""
        self.signal_list.clear()
