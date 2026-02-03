"""Data presenter - handles data loading and management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QObject, pyqtSignal

from data_processor.core.data_loader import DataLoader

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class DataPresenter(QObject):
    """Presenter for data loading and management operations."""

    # Signals
    data_loaded = pyqtSignal(object)  # DataFrame
    load_failed = pyqtSignal(str)  # Error message
    signals_detected = pyqtSignal(list)  # List of signal names
    loading_started = pyqtSignal()
    loading_finished = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        """Initialize the data presenter."""
        super().__init__(parent)
        self.data_loader = DataLoader(use_high_performance=True)
        self.current_data: pd.DataFrame | None = None
        self._file_paths: list[str] = []

    def load_files(self, file_paths: list[str]) -> None:
        """Load data from files."""
        self._file_paths = file_paths
        self.loading_started.emit()

        try:
            df = self._load_files(file_paths)
            if df is not None:
                self.current_data = df
                self.data_loaded.emit(df)
                self._detect_and_emit_signals()
            else:
                self.load_failed.emit("Failed to load files")
        except Exception as e:
            logger.exception("Error loading files")
            self.load_failed.emit(str(e))
        finally:
            self.loading_finished.emit()

    def _load_files(self, file_paths: list[str]) -> pd.DataFrame | None:
        """Load and combine files."""
        if len(file_paths) == 1:
            return self._load_single_file(file_paths[0])
        return self._load_multiple_files(file_paths)

    def _load_single_file(self, file_path: str) -> pd.DataFrame | None:
        """Load a single file."""
        return self.data_loader.load_csv_file(file_path)

    def _load_multiple_files(self, file_paths: list[str]) -> pd.DataFrame | None:
        """Load and combine multiple files."""
        result = self.data_loader.load_multiple_files(file_paths, combine=True)
        if isinstance(result, dict):
            # Combine dictionaries into single DataFrame
            return self.data_loader.combine_dataframes(result)
        return result

    def _detect_and_emit_signals(self) -> None:
        """Detect signals in current data and emit signal."""
        if self.current_data is not None:
            signals = self.get_signals()
            self.signals_detected.emit(signals)

    def get_signals(self) -> list[str]:
        """Get list of numeric signal names from current data."""
        if self.current_data is None:
            return []
        return self.data_loader.get_numeric_signals(self.current_data)

    def get_all_columns(self) -> list[str]:
        """Get all column names from current data."""
        if self.current_data is None:
            return []
        return list(self.current_data.columns)

    def get_data(self) -> pd.DataFrame | None:
        """Get the current DataFrame."""
        return self.current_data

    def set_data(self, df: pd.DataFrame) -> None:
        """Set the current DataFrame."""
        self.current_data = df
        self._detect_and_emit_signals()

    def get_row_count(self) -> int:
        """Get number of rows in current data."""
        if self.current_data is None:
            return 0
        return len(self.current_data)

    def get_column_count(self) -> int:
        """Get number of columns in current data."""
        if self.current_data is None:
            return 0
        return len(self.current_data.columns)

    def has_data(self) -> bool:
        """Check if data is loaded."""
        return self.current_data is not None and not self.current_data.empty

    def clear(self) -> None:
        """Clear current data."""
        self.current_data = None
        self._file_paths = []
