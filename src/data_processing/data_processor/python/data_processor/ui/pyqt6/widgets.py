"""Reusable PyQt6 widgets for Data Processor."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class SignalListWidget(QWidget):
    """Widget for displaying and selecting signals."""

    selectionChanged = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header with count
        header_layout = QHBoxLayout()
        self.title_label = QLabel("Signals")
        self.title_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.count_label = QLabel("(0)")
        self.count_label.setStyleSheet("color: #888;")
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.count_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        # Selection buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_selection)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

    def set_signals(self, signals: list[str]) -> None:
        """Set the list of available signals."""
        self.list_widget.clear()
        for signal in signals:
            self.list_widget.addItem(signal)
        self.count_label.setText(f"({len(signals)})")

    def get_selected_signals(self) -> list[str]:
        """Get currently selected signals."""
        return [item.text() for item in self.list_widget.selectedItems()]

    def _select_all(self) -> None:
        self.list_widget.selectAll()

    def _clear_selection(self) -> None:
        self.list_widget.clearSelection()

    def _on_selection_changed(self) -> None:
        self.selectionChanged.emit(self.get_selected_signals())


class FilterConfigWidget(QWidget):
    """Widget for configuring filter parameters."""

    filterChanged = pyqtSignal(str, dict)

    FILTER_TYPES = [
        "Moving Average",
        "Butterworth Low-pass",
        "Butterworth High-pass",
        "Butterworth Band-pass",
        "Median Filter",
        "Gaussian Filter",
        "Hampel Filter",
        "Z-Score Filter",
        "Savitzky-Golay",
        "FFT Low-pass",
        "FFT High-pass",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Filter type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Filter Type:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(self.FILTER_TYPES)
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        type_layout.addWidget(self.filter_combo, 1)
        layout.addLayout(type_layout)

        # Parameter stack
        self.param_stack = QStackedWidget()
        self._create_param_widgets()
        layout.addWidget(self.param_stack)

    def _create_param_widgets(self) -> None:
        """Create parameter widgets for each filter type."""
        # Moving Average
        ma_widget = QWidget()
        ma_layout = QFormLayout(ma_widget)
        self.ma_window = QSpinBox()
        self.ma_window.setRange(2, 1000)
        self.ma_window.setValue(10)
        ma_layout.addRow("Window Size:", self.ma_window)
        self.param_stack.addWidget(ma_widget)

        # Butterworth Low-pass
        bw_low_widget = QWidget()
        bw_low_layout = QFormLayout(bw_low_widget)
        self.bw_low_order = QSpinBox()
        self.bw_low_order.setRange(1, 10)
        self.bw_low_order.setValue(4)
        self.bw_low_cutoff = QDoubleSpinBox()
        self.bw_low_cutoff.setRange(0.001, 0.999)
        self.bw_low_cutoff.setValue(0.1)
        self.bw_low_cutoff.setDecimals(3)
        bw_low_layout.addRow("Order:", self.bw_low_order)
        bw_low_layout.addRow("Cutoff (normalized):", self.bw_low_cutoff)
        self.param_stack.addWidget(bw_low_widget)

        # Butterworth High-pass
        bw_high_widget = QWidget()
        bw_high_layout = QFormLayout(bw_high_widget)
        self.bw_high_order = QSpinBox()
        self.bw_high_order.setRange(1, 10)
        self.bw_high_order.setValue(4)
        self.bw_high_cutoff = QDoubleSpinBox()
        self.bw_high_cutoff.setRange(0.001, 0.999)
        self.bw_high_cutoff.setValue(0.1)
        self.bw_high_cutoff.setDecimals(3)
        bw_high_layout.addRow("Order:", self.bw_high_order)
        bw_high_layout.addRow("Cutoff (normalized):", self.bw_high_cutoff)
        self.param_stack.addWidget(bw_high_widget)

        # Butterworth Band-pass
        bw_band_widget = QWidget()
        bw_band_layout = QFormLayout(bw_band_widget)
        self.bw_band_order = QSpinBox()
        self.bw_band_order.setRange(1, 10)
        self.bw_band_order.setValue(4)
        self.bw_band_low = QDoubleSpinBox()
        self.bw_band_low.setRange(0.001, 0.999)
        self.bw_band_low.setValue(0.05)
        self.bw_band_low.setDecimals(3)
        self.bw_band_high = QDoubleSpinBox()
        self.bw_band_high.setRange(0.001, 0.999)
        self.bw_band_high.setValue(0.2)
        self.bw_band_high.setDecimals(3)
        bw_band_layout.addRow("Order:", self.bw_band_order)
        bw_band_layout.addRow("Low Cutoff:", self.bw_band_low)
        bw_band_layout.addRow("High Cutoff:", self.bw_band_high)
        self.param_stack.addWidget(bw_band_widget)

        # Median Filter
        median_widget = QWidget()
        median_layout = QFormLayout(median_widget)
        self.median_kernel = QSpinBox()
        self.median_kernel.setRange(3, 101)
        self.median_kernel.setSingleStep(2)
        self.median_kernel.setValue(5)
        median_layout.addRow("Kernel Size (odd):", self.median_kernel)
        self.param_stack.addWidget(median_widget)

        # Gaussian Filter
        gaussian_widget = QWidget()
        gaussian_layout = QFormLayout(gaussian_widget)
        self.gaussian_sigma = QDoubleSpinBox()
        self.gaussian_sigma.setRange(0.1, 100.0)
        self.gaussian_sigma.setValue(1.0)
        self.gaussian_sigma.setDecimals(2)
        gaussian_layout.addRow("Sigma:", self.gaussian_sigma)
        self.param_stack.addWidget(gaussian_widget)

        # Hampel Filter
        hampel_widget = QWidget()
        hampel_layout = QFormLayout(hampel_widget)
        self.hampel_window = QSpinBox()
        self.hampel_window.setRange(3, 101)
        self.hampel_window.setValue(5)
        self.hampel_threshold = QDoubleSpinBox()
        self.hampel_threshold.setRange(0.1, 10.0)
        self.hampel_threshold.setValue(3.0)
        hampel_layout.addRow("Window Size:", self.hampel_window)
        hampel_layout.addRow("Threshold:", self.hampel_threshold)
        self.param_stack.addWidget(hampel_widget)

        # Z-Score Filter
        zscore_widget = QWidget()
        zscore_layout = QFormLayout(zscore_widget)
        self.zscore_threshold = QDoubleSpinBox()
        self.zscore_threshold.setRange(0.1, 10.0)
        self.zscore_threshold.setValue(3.0)
        self.zscore_threshold.setDecimals(1)
        zscore_layout.addRow("Z-Score Threshold:", self.zscore_threshold)
        self.param_stack.addWidget(zscore_widget)

        # Savitzky-Golay
        savgol_widget = QWidget()
        savgol_layout = QFormLayout(savgol_widget)
        self.savgol_window = QSpinBox()
        self.savgol_window.setRange(5, 101)
        self.savgol_window.setSingleStep(2)
        self.savgol_window.setValue(11)
        self.savgol_order = QSpinBox()
        self.savgol_order.setRange(1, 10)
        self.savgol_order.setValue(3)
        savgol_layout.addRow("Window Size (odd):", self.savgol_window)
        savgol_layout.addRow("Polynomial Order:", self.savgol_order)
        self.param_stack.addWidget(savgol_widget)

        # FFT Low-pass
        fft_low_widget = QWidget()
        fft_low_layout = QFormLayout(fft_low_widget)
        self.fft_low_cutoff = QDoubleSpinBox()
        self.fft_low_cutoff.setRange(0.001, 0.999)
        self.fft_low_cutoff.setValue(0.1)
        self.fft_low_cutoff.setDecimals(3)
        fft_low_layout.addRow("Cutoff Frequency:", self.fft_low_cutoff)
        self.param_stack.addWidget(fft_low_widget)

        # FFT High-pass
        fft_high_widget = QWidget()
        fft_high_layout = QFormLayout(fft_high_widget)
        self.fft_high_cutoff = QDoubleSpinBox()
        self.fft_high_cutoff.setRange(0.001, 0.999)
        self.fft_high_cutoff.setValue(0.1)
        self.fft_high_cutoff.setDecimals(3)
        fft_high_layout.addRow("Cutoff Frequency:", self.fft_high_cutoff)
        self.param_stack.addWidget(fft_high_widget)

    def _on_filter_changed(self, filter_type: str) -> None:
        """Handle filter type change."""
        index = self.FILTER_TYPES.index(filter_type)
        self.param_stack.setCurrentIndex(index)
        self.filterChanged.emit(filter_type, self.get_parameters())

    def get_filter_type(self) -> str:
        """Get selected filter type."""
        return self.filter_combo.currentText()

    def get_parameters(self) -> dict:
        """Get current filter parameters."""
        filter_type = self.get_filter_type()

        if filter_type == "Moving Average":
            return {"ma_window": self.ma_window.value()}
        elif filter_type == "Butterworth Low-pass":
            return {
                "bw_order": self.bw_low_order.value(),
                "bw_cutoff": self.bw_low_cutoff.value(),
            }
        elif filter_type == "Butterworth High-pass":
            return {
                "bw_order": self.bw_high_order.value(),
                "bw_cutoff": self.bw_high_cutoff.value(),
            }
        elif filter_type == "Butterworth Band-pass":
            return {
                "bw_order": self.bw_band_order.value(),
                "bw_low": self.bw_band_low.value(),
                "bw_high": self.bw_band_high.value(),
            }
        elif filter_type == "Median Filter":
            return {"median_kernel": self.median_kernel.value()}
        elif filter_type == "Gaussian Filter":
            return {"gaussian_sigma": self.gaussian_sigma.value()}
        elif filter_type == "Hampel Filter":
            return {
                "hampel_window": self.hampel_window.value(),
                "hampel_threshold": self.hampel_threshold.value(),
            }
        elif filter_type == "Z-Score Filter":
            return {"zscore_threshold": self.zscore_threshold.value()}
        elif filter_type == "Savitzky-Golay":
            return {
                "savgol_window": self.savgol_window.value(),
                "savgol_polyorder": self.savgol_order.value(),
            }
        elif filter_type == "FFT Low-pass":
            return {"fft_cutoff": self.fft_low_cutoff.value()}
        elif filter_type == "FFT High-pass":
            return {"fft_cutoff": self.fft_high_cutoff.value()}
        return {}


class StatisticsWidget(QWidget):
    """Widget for displaying signal statistics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Table for statistics
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Signal", "Mean", "Std", "Min", "Max", "Median"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    def update_statistics(self, df: pd.DataFrame, signals: list[str]) -> None:
        """Update statistics display."""
        self.table.setRowCount(len(signals))

        for i, signal in enumerate(signals):
            if signal in df.columns:
                data = df[signal].dropna()
                self.table.setItem(i, 0, QTableWidgetItem(signal))
                self.table.setItem(i, 1, QTableWidgetItem(f"{data.mean():.4f}"))
                self.table.setItem(i, 2, QTableWidgetItem(f"{data.std():.4f}"))
                self.table.setItem(i, 3, QTableWidgetItem(f"{data.min():.4f}"))
                self.table.setItem(i, 4, QTableWidgetItem(f"{data.max():.4f}"))
                self.table.setItem(i, 5, QTableWidgetItem(f"{data.median():.4f}"))

    def clear(self) -> None:
        """Clear statistics."""
        self.table.setRowCount(0)


class DataPreviewWidget(QWidget):
    """Widget for previewing data in a table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info label
        self.info_label = QLabel("No data loaded")
        self.info_label.setStyleSheet("color: #888; padding: 5px;")
        layout.addWidget(self.info_label)

        # Table
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)

    def update_preview(self, df: pd.DataFrame, max_rows: int = 100) -> None:
        """Update data preview."""
        if df is None or df.empty:
            self.info_label.setText("No data loaded")
            self.table.clear()
            return

        rows_shown = min(len(df), max_rows)
        self.info_label.setText(
            f"Showing {rows_shown} of {len(df)} rows, {len(df.columns)} columns"
        )

        display_df = df.head(max_rows)
        self.table.setRowCount(len(display_df))
        self.table.setColumnCount(len(display_df.columns))
        self.table.setHorizontalHeaderLabels(display_df.columns.astype(str).tolist())

        for i in range(len(display_df)):
            for j, _col in enumerate(display_df.columns):
                val = display_df.iloc[i, j]
                text = f"{val:.6g}" if isinstance(val, float) else str(val)
                item = QTableWidgetItem(text)
                self.table.setItem(i, j, item)

    def clear(self) -> None:
        """Clear preview."""
        self.info_label.setText("No data loaded")
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)


class StatusBar(QWidget):
    """Custom status bar with progress indicator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def set_status(self, message: str) -> None:
        """Set status message."""
        self.status_label.setText(message)

    def show_progress(self, value: int = 0, maximum: int = 100) -> None:
        """Show progress bar."""
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        self.progress_bar.setVisible(True)

    def hide_progress(self) -> None:
        """Hide progress bar."""
        self.progress_bar.setVisible(False)

    def set_progress(self, value: int) -> None:
        """Update progress value."""
        self.progress_bar.setValue(value)
