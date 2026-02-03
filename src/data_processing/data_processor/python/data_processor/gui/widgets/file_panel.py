"""File selection panel widget."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    pass


class FilePanel(QWidget):
    """Panel for selecting and loading CSV files."""

    # Signals
    files_selected = pyqtSignal(list)  # List of file paths
    load_requested = pyqtSignal()
    files_cleared = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the file panel."""
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        self._add_title(layout)
        self._add_buttons(layout)
        self._add_file_list(layout)

    def _add_title(self, layout: QVBoxLayout) -> None:
        """Add title label."""
        title = QLabel("File Selection")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

    def _add_buttons(self, layout: QVBoxLayout) -> None:
        """Add action buttons."""
        button_layout = QHBoxLayout()

        self.select_button = QPushButton("Select Files")
        self.clear_button = QPushButton("Clear")
        self.load_button = QPushButton("Load Data")

        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.load_button)

        layout.addLayout(button_layout)

    def _add_file_list(self, layout: QVBoxLayout) -> None:
        """Add file list widget."""
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(100)
        layout.addWidget(self.file_list)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self.select_button.clicked.connect(self._on_select_clicked)
        self.clear_button.clicked.connect(self._on_clear_clicked)
        self.load_button.clicked.connect(self._on_load_clicked)

    def _on_select_clicked(self) -> None:
        """Handle select button click."""
        files = self._open_file_dialog()
        if files:
            self._add_files_to_list(files)
            self.files_selected.emit(files)

    def _on_clear_clicked(self) -> None:
        """Handle clear button click."""
        self.file_list.clear()
        self.files_cleared.emit()

    def _on_load_clicked(self) -> None:
        """Handle load button click."""
        if self.file_list.count() > 0:
            self.load_requested.emit()

    def _open_file_dialog(self) -> list[str]:
        """Open file dialog and return selected files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CSV Files",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        return files

    def _add_files_to_list(self, files: list[str]) -> None:
        """Add files to the list widget."""
        for file_path in files:
            filename = Path(file_path).name
            self.file_list.addItem(filename)

    def get_file_paths(self) -> list[str]:
        """Get list of selected file paths."""
        # Note: This returns display names, actual paths should be stored
        return [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
        ]

    def set_files(self, files: list[str]) -> None:
        """Set the file list."""
        self.file_list.clear()
        self._add_files_to_list(files)
