"""Data export panel widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass

# Available export formats
EXPORT_FORMATS = ["csv", "excel", "parquet", "hdf5", "feather"]


class ExportPanel(QWidget):
    """Panel for exporting processed data."""

    # Signals
    export_requested = pyqtSignal(str)  # Export format

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the export panel."""
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        self._add_title(layout)
        self._add_format_selector(layout)
        self._add_export_button(layout)

    def _add_title(self, layout: QVBoxLayout) -> None:
        """Add title label."""
        title = QLabel("Export Data")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

    def _add_format_selector(self, layout: QVBoxLayout) -> None:
        """Add format selector."""
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Format:"))

        self.format_combo = QComboBox()
        self.format_combo.addItems(EXPORT_FORMATS)
        selector_layout.addWidget(self.format_combo)

        layout.addLayout(selector_layout)

    def _add_export_button(self, layout: QVBoxLayout) -> None:
        """Add export button."""
        self.export_button = QPushButton("Export")
        layout.addWidget(self.export_button)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.export_button.clicked.connect(self._on_export_clicked)

    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        export_format = self.get_export_format()
        self.export_requested.emit(export_format)

    def get_export_format(self) -> str:
        """Get selected export format."""
        return self.format_combo.currentText()

    def set_export_format(self, fmt: str) -> None:
        """Set the export format."""
        index = self.format_combo.findText(fmt)
        if index >= 0:
            self.format_combo.setCurrentIndex(index)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the export button."""
        self.export_button.setEnabled(enabled)
