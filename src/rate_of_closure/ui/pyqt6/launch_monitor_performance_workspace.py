"""Dispersion, target scoring, and trusted-session trends for PyQt."""

from __future__ import annotations

import json

import pandas as pd
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.atomic_text_files import write_utf8_text_atomic
from rate_of_closure.launch_monitor_analysis import numeric_columns
from rate_of_closure.launch_monitor_performance import (
    DispersionRequest,
    DispersionResult,
    ScoreResult,
    StrokesGainedRequest,
    TargetErrorRequest,
    TrendRequest,
    TrendResult,
    analyze_dispersion,
    analyze_session_trend,
    calculate_strokes_gained,
    calculate_target_error,
)
from rate_of_closure.launch_monitor_workspace import dataset_reference_for_frame
from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas
from rate_of_closure.ui.pyqt6.launch_monitor_performance_files import (
    load_performance_settings,
    performance_document,
)
from rate_of_closure.ui.pyqt6.launch_monitor_performance_presenter import (
    present_value_error,
)


class LaunchMonitorPerformanceWorkspace(QWidget):
    """Present descriptive metrics without inferring identity or baseline state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._source_name = "Unloaded"
        self._dispersion: DispersionResult | None = None
        self._proxy: ScoreResult | None = None
        self._trend: TrendResult | None = None
        self._build_ui()

    def _combo(self, name: str) -> QComboBox:
        combo = QComboBox()
        combo.setAccessibleName(name)
        combo.setToolTip(name)
        return combo

    def _build_ui(self) -> None:
        self.carry_combo = self._combo("Dispersion carry column")
        self.lateral_combo = self._combo("Dispersion lateral column")
        self.carry_unit = self._combo("Carry source unit")
        self.lateral_unit = self._combo("Lateral source unit")
        self.carry_unit.addItems(["yd", "m"])
        self.lateral_unit.addItems(["yd", "m"])
        self.target_distance = QDoubleSpinBox()
        self.target_distance.setRange(1, 1000)
        self.target_distance.setValue(150)
        self.dispersion_button = QPushButton("Analyze Dispersion & Target Error")
        self.dispersion_status = QLabel("Select carry and lateral columns.")
        self.dispersion_status.setWordWrap(True)
        figure = Figure(figsize=(6.5, 3.1), layout="constrained")
        self.canvas = LifecycleSafeFigureCanvas(figure)
        self.axes = figure.add_subplot(111)
        self.axes.set_xlabel("Carry (yd)")
        self.axes.set_ylabel("Lateral (yd; left − / right +)")

        self.before_combo = self._combo("Expected strokes before column")
        self.after_combo = self._combo("Expected strokes after column")
        self.baseline_url = QLineEdit()
        self.strokes_button = QPushButton("Calculate Strokes Gained")
        self.strokes_status = QLabel(
            "Unavailable: provide expected-stroke columns and a cited baseline source."
        )
        self.strokes_status.setWordWrap(True)

        self.player_combo = self._combo("Trusted player identity column")
        self.session_combo = self._combo("Trusted session identity column")
        self.order_combo = self._combo("Explicit session order column")
        self.metric_combo = self._combo("Session trend metric")
        self.player_attest = QCheckBox("Player identity is supplied and trusted")
        self.session_attest = QCheckBox(
            "Session identity and order are supplied and trusted"
        )
        self.trend_button = QPushButton("Run Session Trend")
        self.trend_status = QLabel(
            "Unavailable until both identity attestations are checked."
        )
        self.trend_status.setWordWrap(True)

        self.save_button = QPushButton("Save Performance Analysis...")
        self.load_button = QPushButton("Load Performance Analysis...")
        self.export_plot_button = QPushButton("Export Plot...")
        self.export_data_button = QPushButton("Export Backing Data...")
        described = (
            (self.target_distance, "Target distance in yards for radial target error"),
            (
                self.dispersion_button,
                "Calculate yards-left/right dispersion and radial target error",
            ),
            (self.before_combo, "Expected strokes before column"),
            (self.after_combo, "Expected strokes after column"),
            (self.baseline_url, "HTTP(S) source for the expected-strokes baseline"),
            (self.strokes_button, "Calculate source-backed strokes gained"),
            (self.player_attest, "Explicit trusted player identity attestation"),
            (
                self.session_attest,
                "Explicit trusted session identity and order attestation",
            ),
            (self.trend_button, "Calculate session and cumulative means"),
            (
                self.save_button,
                "Save settings, results, formulas, and dataset fingerprint",
            ),
            (
                self.load_button,
                "Reload settings only against the matching dataset fingerprint",
            ),
            (self.export_plot_button, "Export the unit-labeled dispersion plot"),
            (self.export_data_button, "Export every backing input row as CSV"),
        )
        for control, description in described:
            control.setAccessibleName(description)
            control.setToolTip(description)

        tabs = QTabWidget()
        tabs.addTab(self._dispersion_page(), "Dispersion")
        tabs.addTab(self._strokes_page(), "Strokes Gained")
        tabs.addTab(self._trend_page(), "Session Trends")
        buttons = QHBoxLayout()
        for button in (
            self.save_button,
            self.load_button,
            self.export_plot_button,
            self.export_data_button,
        ):
            buttons.addWidget(button)
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Performance Analytics — formulas and availability are shown "
                "in each tab"
            )
        )
        layout.addWidget(tabs)
        layout.addLayout(buttons)
        self.dispersion_button.clicked.connect(self.run_dispersion_safely)
        self.strokes_button.clicked.connect(self.run_strokes_safely)
        self.trend_button.clicked.connect(self.run_trend_safely)
        self.player_attest.toggled.connect(self._refresh_enabled)
        self.session_attest.toggled.connect(self._refresh_enabled)
        self.baseline_url.textChanged.connect(self._refresh_enabled)
        self.before_combo.currentTextChanged.connect(self._refresh_enabled)
        self.after_combo.currentTextChanged.connect(self._refresh_enabled)
        self.save_button.clicked.connect(self.save_dialog)
        self.load_button.clicked.connect(self.load_dialog)
        self.export_plot_button.clicked.connect(self.export_plot_dialog)
        self.export_data_button.clicked.connect(self.export_data_dialog)
        self._refresh_enabled()

    def _dispersion_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.addRow("Carry:", self.carry_combo)
        form.addRow("Carry unit:", self.carry_unit)
        form.addRow("Lateral:", self.lateral_combo)
        form.addRow("Lateral unit:", self.lateral_unit)
        form.addRow("Target distance (yd):", self.target_distance)
        form.addRow(self.dispersion_button)
        form.addRow(self.dispersion_status)
        form.addRow(self.canvas)
        return page

    def _strokes_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.addRow("Expected before:", self.before_combo)
        form.addRow("Expected after:", self.after_combo)
        form.addRow("Baseline source URL:", self.baseline_url)
        form.addRow(self.strokes_button)
        form.addRow(self.strokes_status)
        return page

    def _trend_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.addRow("Player:", self.player_combo)
        form.addRow("Session:", self.session_combo)
        form.addRow("Session order:", self.order_combo)
        form.addRow("Metric:", self.metric_combo)
        form.addRow(self.player_attest)
        form.addRow(self.session_attest)
        form.addRow(self.trend_button)
        form.addRow(self.trend_status)
        return page

    def set_dataset(self, frame: pd.DataFrame, source_name: str) -> None:
        """Bind controls to a new retained dataset and invalidate results."""

        self._frame = frame
        self._source_name = source_name
        all_columns = sorted(str(column) for column in frame.columns)
        numbers = numeric_columns(frame)
        for combo, values in (
            (self.carry_combo, numbers),
            (self.lateral_combo, numbers),
            (self.before_combo, numbers),
            (self.after_combo, numbers),
            (self.player_combo, all_columns),
            (self.session_combo, all_columns),
            (self.order_combo, numbers),
            (self.metric_combo, numbers),
        ):
            combo.clear()
            combo.addItem("")
            combo.addItems(values)
        self.player_attest.setChecked(False)
        self.session_attest.setChecked(False)
        self._dispersion = self._proxy = self._trend = None
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        self.trend_button.setEnabled(
            self.player_attest.isChecked() and self.session_attest.isChecked()
        )
        strokes_ready = bool(
            self.before_combo.currentText()
            and self.after_combo.currentText()
            and self.baseline_url.text().strip()
        )
        self.strokes_button.setEnabled(strokes_ready)

    def run_dispersion(self) -> tuple[DispersionResult, ScoreResult]:
        request = DispersionRequest(
            self.lateral_combo.currentText(),
            self.carry_combo.currentText(),
            self.lateral_unit.currentText(),
            self.carry_unit.currentText(),
        )
        dispersion = analyze_dispersion(self._frame, request)
        proxy = calculate_target_error(
            self._frame,
            TargetErrorRequest(
                request.carry_column,
                request.lateral_column,
                request.carry_unit,
                request.lateral_unit,
                self.target_distance.value(),
            ),
        )
        self._dispersion, self._proxy = dispersion, proxy
        self.axes.clear()
        self.axes.scatter(
            [point.carry_yards for point in dispersion.points],
            [point.lateral_yards for point in dispersion.points],
            alpha=0.7,
        )
        self.axes.axhline(0, color="#94a3b8", linewidth=1)
        self.axes.set_xlabel("Carry (yd)")
        self.axes.set_ylabel("Lateral (yd; left − / right +)")
        self.axes.set_title("Shot dispersion relative to target line")
        self.canvas.draw_idle()
        status = (
            f"{dispersion.left_count} yards-left shots · "
            f"{dispersion.right_count} yards-right shots · radial target error "
            f"{proxy.mean:.2f} yd. {dispersion.formula}"
        )
        self.dispersion_status.setText(status)
        return dispersion, proxy

    def run_dispersion_safely(self) -> None:
        present_value_error(self, "Dispersion Unavailable", self.run_dispersion)

    def run_strokes(self) -> ScoreResult:
        result = calculate_strokes_gained(
            self._frame,
            StrokesGainedRequest(
                self.before_combo.currentText(),
                self.after_combo.currentText(),
                self.baseline_url.text().strip(),
            ),
        )
        self.strokes_status.setText(
            f"Mean {result.mean:.3f} strokes. {result.formula}. "
            f"Source: {result.source_url}"
        )
        return result

    def run_strokes_safely(self) -> None:
        present_value_error(self, "Strokes Gained Unavailable", self.run_strokes)

    def run_trend(self) -> TrendResult:
        result = analyze_session_trend(
            self._frame,
            TrendRequest(
                self.metric_combo.currentText(),
                self.session_combo.currentText(),
                self.order_combo.currentText(),
                self.player_combo.currentText(),
                self.player_attest.isChecked(),
                self.session_attest.isChecked(),
            ),
        )
        self._trend = result
        self.trend_status.setText(
            f"{len(result.points)} player-session points. {result.formula}"
        )
        return result

    def run_trend_safely(self) -> None:
        present_value_error(self, "Trend Unavailable", self.run_trend)

    def _document(self) -> dict[str, object]:
        reference = dataset_reference_for_frame(self._frame, self._source_name)
        settings = {
            "carry": self.carry_combo.currentText(),
            "lateral": self.lateral_combo.currentText(),
            "carry_unit": self.carry_unit.currentText(),
            "lateral_unit": self.lateral_unit.currentText(),
            "target_yards": self.target_distance.value(),
        }
        return performance_document(
            reference, settings, self._dispersion, self._proxy, self._trend
        )

    def save_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Performance Analysis",
            "performance.lmanalysis.json",
            "JSON (*.json)",
        )
        if selected:
            write_utf8_text_atomic(
                json.dumps(self._document(), indent=2),
                selected,
                document_name="performance analysis",
            )

    def load_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Load Performance Analysis", "", "JSON (*.json)"
        )
        if not selected:
            return
        try:
            reference = dataset_reference_for_frame(self._frame, self._source_name)
            settings = load_performance_settings(selected, reference.sha256)
            self.carry_combo.setCurrentText(settings["carry"])
            self.lateral_combo.setCurrentText(settings["lateral"])
            self.carry_unit.setCurrentText(settings["carry_unit"])
            self.lateral_unit.setCurrentText(settings["lateral_unit"])
            self.target_distance.setValue(float(settings["target_yards"]))
            self.dispersion_status.setText(
                "Saved settings restored; rerun to regenerate results."
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.warning(self, "Analysis Not Loaded", str(error))

    def export_plot_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Unit-Labeled Plot",
            "dispersion.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if selected:
            self.canvas.figure.savefig(selected)

    def export_data_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export Backing Data", "performance-backing.csv", "CSV (*.csv)"
        )
        if selected:
            self._frame.to_csv(selected, index=False)


__all__ = ["LaunchMonitorPerformanceWorkspace"]
