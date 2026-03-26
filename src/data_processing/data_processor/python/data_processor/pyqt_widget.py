"""PyQt6 widget wrapper for the Data Processor core."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data_processor.core.data_loader import DataLoader
from data_processor.core.signal_processor import SignalProcessor
from data_processor.models.processing_config import FilterConfig

logger = logging.getLogger(__name__)


class DataProcessorWidget(QWidget):
    """A PyQt6 widget that wraps the Data Processor core functionality.

    This widget provides a UI for:
    1. Loading CSV files.
    2. Selecting signals/columns.
    3. Applying filters to selected signals.
    4. Viewing the processed results.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.data_loader = DataLoader()
        self.signal_processor = SignalProcessor()
        self.current_df: pd.DataFrame | None = None
        self.processed_df: pd.DataFrame | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # File Loading Section
        file_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load CSV")
        self.load_btn.clicked.connect(self.load_file)
        self.file_label = QLabel("No file loaded")
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(self.file_label)
        layout.addLayout(file_layout)

        # Signal Selection Section
        layout.addWidget(QLabel("Select Signals:"))
        self.signal_list = QListWidget()
        self.signal_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        layout.addWidget(self.signal_list)

        # Filter Configuration Section
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        # Populate with allowed filters from FilterConfig models
        # Note: We hardcode some common ones for now as
        # models.processing_config._ALLOWED_FILTERS is private
        self.filter_combo.addItems(
            [
                "Moving Average",
                "Butterworth Low-pass",
                "Butterworth High-pass",
                "Median Filter",
                "Savitzky-Golay",
                "Z-Score Filter",
            ]
        )
        filter_layout.addWidget(self.filter_combo)

        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_data)
        filter_layout.addWidget(self.process_btn)
        layout.addLayout(filter_layout)

        # Results Section
        layout.addWidget(QLabel("Results Preview:"))
        self.result_table = QTableWidget()
        layout.addWidget(self.result_table)

    def load_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                self.current_df = self.data_loader.load_csv_file(file_path)
                if self.current_df is not None:
                    self.file_label.setText(Path(file_path).name)
                    self._populate_signals()
                    self._update_table(self.current_df)
                    self.processed_df = None
                else:
                    QMessageBox.warning(self, "Error", "Failed to load file.")
            except (PermissionError, OSError) as e:
                logger.error(f"Error loading file: {e}")
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def _populate_signals(self) -> None:
        self.signal_list.clear()
        if self.current_df is not None:
            numeric_signals = self.data_loader.get_numeric_signals(self.current_df)
            self.signal_list.addItems(numeric_signals)

    def process_data(self) -> None:
        if self.current_df is None:
            QMessageBox.warning(self, "Warning", "Please load a file first.")
            return

        selected_items = self.signal_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one signal.")
            return

        selected_signals = [item.text() for item in selected_items]
        filter_type = self.filter_combo.currentText()

        # Default parameters for now - in a real app these would be configurable via UI
        params: dict[str, Any] = {}
        if filter_type == "Moving Average":
            params = {"ma_window": 5}
        elif "Butterworth" in filter_type:
            params = {"bw_cutoff": 0.1, "bw_order": 4}
        elif filter_type == "Median Filter":
            params = {"median_kernel": 3}
        elif filter_type == "Savitzky-Golay":
            params = {"savgol_window": 5, "savgol_polyorder": 2}
        elif filter_type == "Z-Score Filter":
            params = {"zscore_threshold": 3.0}

        try:
            config = FilterConfig(filter_type=filter_type, parameters=params)

            # Filter only selected signals
            df_to_process = self.current_df[selected_signals].copy()

            # The SignalProcessor.apply_filter returns a dataframe with filtered signals
            # Note: VectorizedFilterEngine usually returns same columns.
            # If we want to preserve other columns, we should merge.
            # For this widget, showing just the processed result is fine.

            processed_subset = self.signal_processor.apply_filter(df_to_process, config)

            self.processed_df = processed_subset
            self._update_table(self.processed_df)
            QMessageBox.information(self, "Success", "Processing complete.")

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error processing data: {e}")
            QMessageBox.critical(self, "Error", f"Processing failed: {e}")

    def _update_table(self, df: pd.DataFrame) -> None:
        if not (df is not None):
            raise ValueError("df must be provided")
        self.result_table.clear()
        # Show max 100 rows for preview
        self.result_table.setRowCount(min(100, len(df)))
        self.result_table.setColumnCount(len(df.columns))
        self.result_table.setHorizontalHeaderLabels(df.columns.astype(str))

        for i in range(min(100, len(df))):
            for j, _col in enumerate(df.columns):
                val = df.iloc[i, j]
                self.result_table.setItem(i, j, QTableWidgetItem(str(val)))
