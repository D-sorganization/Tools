"""PyQt6 widget wrapper for the Data Processor core."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from data_processor.core.data_loader import DataLoader
from data_processor.core.signal_processor import SignalProcessor
from data_processor.models.processing_config import FilterConfig
from data_processor.ui.async_workers import DataLoadResult, DataLoadWorker
from data_processor.ui.async_workers import ProcessingWorker as AsyncProcessingWorker

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
        self._load_worker: DataLoadWorker | None = None
        self._process_worker: AsyncProcessingWorker | None = None

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

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Results Section
        layout.addWidget(QLabel("Results Preview:"))
        self.result_table = QTableWidget()
        layout.addWidget(self.result_table)

    def load_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self._start_load_file(file_path)

    def _start_load_file(self, file_path: str) -> None:
        self._set_busy(True)
        self.file_label.setText(f"Loading {Path(file_path).name}...")
        worker = DataLoadWorker([file_path], self.data_loader)
        self._load_worker = worker
        worker.result_ready.connect(
            lambda result: self._on_file_loaded(file_path, result)
        )
        worker.error.connect(self._on_file_load_error)
        worker.finished.connect(self._on_load_finished)
        worker.start()

    def _on_file_loaded(self, file_path: str, result: DataLoadResult) -> None:
        self.current_df = result.data
        self.file_label.setText(Path(file_path).name)
        self._populate_signals()
        self._update_table(self.current_df)
        self.processed_df = None

    def _on_file_load_error(self, message: str) -> None:
        logger.error("Error loading file: %s", message)
        self.file_label.setText("No file loaded")
        QMessageBox.critical(self, "Error", f"An error occurred: {message}")

    def _on_load_finished(self) -> None:
        if self._load_worker is not None:
            self._load_worker.deleteLater()
        self._load_worker = None
        self._set_busy(False)

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
            FilterConfig(filter_type=filter_type, parameters=params)
            df_to_process = self.current_df[selected_signals].copy()
        except (KeyError, ValueError, TypeError) as e:
            logger.error("Error preparing processing job: %s", e)
            QMessageBox.critical(self, "Error", f"Processing failed: {e}")
            return

        self._set_busy(True)
        worker = AsyncProcessingWorker(
            "filter",
            df_to_process,
            self.signal_processor,
            {"filter_type": filter_type, "parameters": params},
        )
        self._process_worker = worker
        worker.result_ready.connect(self._on_processing_complete)
        worker.error.connect(self._on_processing_error)
        worker.finished.connect(self._on_processing_finished)
        worker.start()

    def _on_processing_complete(self, processed_subset: pd.DataFrame) -> None:
        self.processed_df = processed_subset
        self._update_table(self.processed_df)
        QMessageBox.information(self, "Success", "Processing complete.")

    def _on_processing_error(self, message: str) -> None:
        logger.error("Error processing data: %s", message)
        QMessageBox.critical(self, "Error", f"Processing failed: {message}")

    def _on_processing_finished(self) -> None:
        if self._process_worker is not None:
            self._process_worker.deleteLater()
        self._process_worker = None
        self._set_busy(False)

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802 - Qt API
        """Join background workers before the widget is destroyed.

        A ``QThread`` that outlives its owner is destroyed while running,
        which aborts the process, or blocks interpreter shutdown while Qt
        waits on it. Either way the failure surfaces far from its cause, so
        the widget refuses to close with live workers unaccounted for.
        """
        self.shutdown_workers()
        super().closeEvent(event)

    def shutdown_workers(self, timeout_ms: int = 5000) -> None:
        """Block until background workers exit; terminate as a last resort.

        Preconditions: ``timeout_ms`` must be positive.
        Postconditions: both worker attributes are ``None`` and no worker
        thread started by this widget is still running.
        """
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        for name in ("_load_worker", "_process_worker"):
            worker = getattr(self, name)
            if worker is None:
                continue
            if worker.isRunning() and not worker.wait(timeout_ms):
                logger.error(
                    "%s did not finish within %d ms; terminating it to avoid "
                    "aborting the process at interpreter shutdown",
                    name,
                    timeout_ms,
                )
                worker.terminate()
                worker.wait(1000)
            setattr(self, name, None)

    def _set_busy(self, busy: bool) -> None:
        self.load_btn.setEnabled(not busy)
        self.process_btn.setEnabled(not busy and self.current_df is not None)
        self.progress_bar.setVisible(busy)

    def _update_table(self, df: pd.DataFrame) -> None:
        if df is None:
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
