# mypy: ignore-errors
# ruff: noqa: E501
import csv
import logging

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# Try to import filters from signal_toolkit, with safe fallbacks
try:
    from signal_toolkit.filters import (
        create_moving_average_filter,
        create_savgol_filter,
    )
except ImportError:
    create_moving_average_filter = None
    create_savgol_filter = None

logger = logging.getLogger("p1am_control.desktop.trends")


class TrendsTab(QWidget):
    """Real-time trending and data filtering tab."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("trends_tab")

        # State variables
        self.is_playing = True
        self.history_len_sec = 60  # Default 60 seconds
        self.sampling_rate = 10  # 10Hz

        # Buffers for raw time-series data
        self.time_buffer = []  # relative time seconds
        self.value_buffer = []  # raw values
        self.absolute_time_buffer = []  # system time strings or floats

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Control Panel
        controls_layout = QHBoxLayout()

        # Tag Selector
        controls_layout.addWidget(QLabel("Tag Select:", self))
        self.tag_combo = QComboBox(self)
        self.tag_combo.addItems(
            [
                "Tag 1: Drying Hopper Level",
                "Tag 2: Feed Valve CV",
                "Tag 3: Pyrolysis Temp",
                "Tag 4: Air Valve CV",
                "Tag 5: Combustion Temp",
                "Tag 6: Quench Valve CV",
                "Tag 7: Reduction Temp",
                "Tag 8: Flare Valve CV",
                "Tag 9: CPU Temperature",
                "Tag 10: Cycle Time",
            ]
        )
        # Allow custom index mapping
        self.tag_mapping = {
            0: 1,  # Combobox index 0 maps to Tag 1
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
        }
        controls_layout.addWidget(self.tag_combo)

        # Play / Pause
        self.btn_play_pause = QPushButton("Pause", self)
        self.btn_play_pause.setCheckable(True)
        self.btn_play_pause.clicked.connect(self._toggle_play)
        controls_layout.addWidget(self.btn_play_pause)

        # Zoom Reset
        self.btn_reset_zoom = QPushButton("Reset Zoom", self)
        self.btn_reset_zoom.clicked.connect(self._reset_zoom)
        controls_layout.addWidget(self.btn_reset_zoom)

        # Range Selector
        controls_layout.addWidget(QLabel("Range (sec):", self))
        self.spin_range = QSpinBox(self)
        self.spin_range.setRange(10, 600)
        self.spin_range.setValue(60)
        self.spin_range.setSuffix("s")
        self.spin_range.valueChanged.connect(self._on_range_changed)
        controls_layout.addWidget(self.spin_range)

        # Exports
        self.btn_export_csv = QPushButton("Export CSV", self)
        self.btn_export_csv.clicked.connect(self._export_csv)
        self.btn_export_svg = QPushButton("Export SVG", self)
        self.btn_export_svg.clicked.connect(self._export_svg)
        controls_layout.addWidget(self.btn_export_csv)
        controls_layout.addWidget(self.btn_export_svg)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Filters Checkboxes Panel
        filters_layout = QHBoxLayout()
        filters_layout.addWidget(QLabel("Filters:", self))

        self.chk_ma = QCheckBox("Moving Average", self)
        self.chk_savgol = QCheckBox("Savitzky-Golay", self)
        self.chk_fft_low = QCheckBox("FFT Lowpass", self)
        self.chk_fft_notch = QCheckBox("FFT Notch (60Hz)", self)

        # Exclusivity or multi-filter options
        filters_layout.addWidget(self.chk_ma)
        filters_layout.addWidget(self.chk_savgol)
        filters_layout.addWidget(self.chk_fft_low)
        filters_layout.addWidget(self.chk_fft_notch)

        filters_layout.addStretch()
        layout.addLayout(filters_layout)

        # PyQtGraph Plot Widget
        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setBackground("#15181e")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel("left", "Value")
        self.plot_widget.setLabel("bottom", "Time (seconds)")

        # Legend
        self.plot_widget.addLegend()

        # Curves
        self.raw_curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#7f8c8d", width=1, style=Qt.PenStyle.DashLine),
            name="Raw Signal",
        )
        self.filtered_curve = self.plot_widget.plot(
            pen=pg.mkPen(color="#00f2fe", width=2), name="Filtered Signal"
        )

        layout.addWidget(self.plot_widget)

    def _toggle_play(self, checked: bool) -> None:
        self.is_playing = not checked
        self.btn_play_pause.setText("Play" if checked else "Pause")
        logger.info(f"Trends play/pause set to: {'PAUSED' if checked else 'PLAYING'}")

    def _reset_zoom(self) -> None:
        self.plot_widget.autoRange()

    def _on_range_changed(self, val: int) -> None:
        self.history_len_sec = val

    def get_selected_tag_id(self) -> int:
        idx = self.tag_combo.currentIndex()
        return self.tag_mapping.get(idx, 1)

    def add_telemetry_point(self, timestamp: float, tags: list[float]) -> None:
        """Called by main window thread at 10Hz to feed telemetry data."""
        if not self.is_playing:
            return

        tag_id = self.get_selected_tag_id()
        if tag_id >= len(tags):
            return

        val = tags[tag_id]

        self.time_buffer.append(timestamp)
        self.value_buffer.append(val)
        self.absolute_time_buffer.append(timestamp)

        # Slice buffers to only keep maximum configured range + some headroom
        max_samples = (self.history_len_sec + 10) * self.sampling_rate
        if len(self.time_buffer) > max_samples:
            self.time_buffer = self.time_buffer[-max_samples:]
            self.value_buffer = self.value_buffer[-max_samples:]
            self.absolute_time_buffer = self.absolute_time_buffer[-max_samples:]

        self._update_plot()

    def _update_plot(self) -> None:
        if not self.time_buffer:
            return

        # Prepare arrays
        times = np.array(self.time_buffer)
        raw_vals = np.array(self.value_buffer)

        # Relative time coordinates (scrolling window, e.g. -60s to 0s)
        if len(times) > 1:
            rel_times = times - times[-1]
        else:
            rel_times = np.array([0.0])

        # Filter the relative time window we want to view
        cutoff_time = -self.history_len_sec
        visible_mask = rel_times >= cutoff_time

        view_times = rel_times[visible_mask]
        view_raw = raw_vals[visible_mask]

        if len(view_raw) == 0:
            return

        # Apply digital filters if checked
        filtered_vals = view_raw.copy()

        # 1. Moving Average filter
        if self.chk_ma.isChecked():
            window = 5
            if create_moving_average_filter:
                ma_func = create_moving_average_filter(window)
                filtered_vals = ma_func(filtered_vals)
            else:
                kernel = np.ones(window) / window
                filtered_vals = np.convolve(filtered_vals, kernel, mode="same")

        # 2. Savitzky-Golay filter
        if self.chk_savgol.isChecked() and len(filtered_vals) >= 11:
            if create_savgol_filter:
                savgol_func = create_savgol_filter(11, 3)
                filtered_vals = savgol_func(filtered_vals)
            else:
                from scipy.signal import savgol_filter

                filtered_vals = savgol_filter(filtered_vals, 11, 3)

        # 3. FFT Lowpass filter
        if self.chk_fft_low.isChecked() and len(filtered_vals) > 4:
            # Clean FFT lowpass: take rfft, zero out frequencies above a cutoff, irfft
            rfft_coeffs = np.fft.rfft(filtered_vals)
            cutoff_freq_idx = max(
                2, int(len(rfft_coeffs) * 0.25)
            )  # Keep lower 25% of frequencies
            rfft_coeffs[cutoff_freq_idx:] = 0.0
            filtered_vals = np.fft.irfft(rfft_coeffs, len(filtered_vals))

        # 4. FFT Notch filter (around simulated 60Hz noise or similar high freq)
        if self.chk_fft_notch.isChecked() and len(filtered_vals) > 8:
            rfft_coeffs = np.fft.rfft(filtered_vals)
            # Notch out index around 70-80% of frequency spectrum
            notch_start = int(len(rfft_coeffs) * 0.6)
            notch_end = int(len(rfft_coeffs) * 0.8)
            rfft_coeffs[notch_start:notch_end] = 0.0
            filtered_vals = np.fft.irfft(rfft_coeffs, len(filtered_vals))

        # Render curves
        self.raw_curve.setData(view_times, view_raw)
        self.filtered_curve.setData(view_times, filtered_vals)

        # Set view window
        self.plot_widget.setXRange(cutoff_time, 0)

    def _export_csv(self) -> None:
        if not self.time_buffer:
            QMessageBox.warning(
                self, "Export Failed", "No trend data available to export."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Trend Data", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Raw Value"])
                for t, v in zip(
                    self.absolute_time_buffer, self.value_buffer, strict=False
                ):
                    writer.writerow([t, v])
            QMessageBox.information(
                self, "Export Successful", f"Trend data saved to {file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save CSV file: {e}")

    def _export_svg(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Trend Plot as SVG", "", "SVG Files (*.svg)"
        )
        if not file_path:
            return

        try:
            exporter = pyqtgraph.exporters.SVGExporter(self.plot_widget.plotItem)
            exporter.export(file_path)
            QMessageBox.information(
                self, "Export Successful", f"SVG plot saved to {file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save SVG plot: {e}")
