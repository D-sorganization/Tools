"""PyQt presentation for identity-safe longitudinal player analysis."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.launch_monitor_analysis import numeric_columns
from rate_of_closure.launch_monitor_longitudinal import (
    LongitudinalRequest,
    LongitudinalResult,
    analyze_longitudinal_performance,
)
from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas


def _unit(column: str) -> str:
    lowered = column.lower()
    for token, unit in (
        ("speed", "unit unknown"),
        ("angle", "deg"),
        ("spin", "rpm"),
        ("distance", "unit unknown"),
        ("_yd", "yd"),
        ("_m", "m"),
    ):
        if token in lowered:
            return unit
    return "unit unknown"


def _text(value: object) -> str:
    return (
        "—"
        if value is None
        else f"{value:.4g}"
        if isinstance(value, float)
        else str(value)
    )


class LaunchMonitorLongitudinalWidget(QWidget):
    """Render session uncertainty, player slopes, and population synthesis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self.result: LongitudinalResult | None = None
        self._build_ui()

    def _combo(self, name: str) -> QComboBox:
        combo = QComboBox()
        combo.setAccessibleName(name)
        combo.setToolTip(name)
        return combo

    def _build_ui(self) -> None:
        self.player_combo = self._combo("Longitudinal player identity column")
        self.session_combo = self._combo("Longitudinal session identity column")
        self.order_combo = self._combo("Longitudinal session order column")
        self.metric_combo = self._combo("Longitudinal performance metric")
        self.player_attest = QCheckBox("Player identity is supplied and trusted")
        self.session_attest = QCheckBox(
            "Session identity and order are supplied and trusted"
        )
        self.higher_better = QCheckBox("Higher metric values represent improvement")
        self.higher_better.setChecked(True)
        self.min_sessions = QSpinBox()
        self.min_sessions.setRange(3, 10_000)
        self.min_sessions.setValue(3)
        self.calculate_button = QPushButton("Run Longitudinal Inference")
        self.save_plot_button = QPushButton("Save Longitudinal Plot...")
        self.status = QLabel(
            "Unavailable until both identity attestations are checked."
        )
        self.status.setWordWrap(True)
        for control, name in (
            (self.player_attest, "Explicit player identity attestation"),
            (self.session_attest, "Explicit session identity and order attestation"),
            (self.higher_better, "Whether higher metric values represent improvement"),
            (self.min_sessions, "Minimum sessions per player for slope synthesis"),
            (self.calculate_button, "Estimate session, player, and population trends"),
            (self.save_plot_button, "Save the unit-labelled longitudinal plot"),
        ):
            control.setAccessibleName(name)
            control.setToolTip(name)
        figure = Figure(figsize=(7, 3), layout="constrained")
        self.canvas = LifecycleSafeFigureCanvas(figure)
        self.canvas.setAccessibleName("Longitudinal session trend plot")
        self.axes = figure.add_subplot(111)
        self.table = QTableWidget()
        self.table.setAccessibleName("Player longitudinal estimates")
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._install_layout()
        self.calculate_button.clicked.connect(self.calculate_safely)
        self.save_plot_button.clicked.connect(self.save_plot_dialog)
        self.player_attest.toggled.connect(self._refresh_enabled)
        self.session_attest.toggled.connect(self._refresh_enabled)
        self._refresh_enabled()

    def _install_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Player:", self.player_combo)
        form.addRow("Session:", self.session_combo)
        form.addRow("Session order:", self.order_combo)
        form.addRow("Metric:", self.metric_combo)
        form.addRow(self.player_attest)
        form.addRow(self.session_attest)
        form.addRow(self.higher_better)
        form.addRow("Minimum sessions/player:", self.min_sessions)
        form.addRow(self.calculate_button)
        form.addRow(self.save_plot_button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(self.canvas)
        layout.addWidget(self.table)

    def set_dataset(self, frame: pd.DataFrame) -> None:
        self._frame = frame
        columns = sorted(str(column) for column in frame.columns)
        numeric = numeric_columns(frame)
        for combo, values in (
            (self.player_combo, columns),
            (self.session_combo, columns),
            (self.order_combo, numeric),
            (self.metric_combo, numeric),
        ):
            combo.clear()
            combo.addItem("")
            combo.addItems(values)
        self.player_attest.setChecked(False)
        self.session_attest.setChecked(False)
        self.result = None
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        self.calculate_button.setEnabled(
            self.player_attest.isChecked() and self.session_attest.isChecked()
        )
        self.save_plot_button.setEnabled(self.result is not None)

    def calculate(self) -> LongitudinalResult:
        request = LongitudinalRequest(
            self.metric_combo.currentText(),
            self.session_combo.currentText(),
            self.order_combo.currentText(),
            self.player_combo.currentText(),
            self.player_attest.isChecked(),
            self.session_attest.isChecked(),
            self.higher_better.isChecked(),
            0.95,
            self.min_sessions.value(),
        )
        result = analyze_longitudinal_performance(self._frame, request)
        self.result = result
        self._plot(result)
        self._table(result)
        population = result.population
        self.status.setText(
            f"{len(result.session_points)} sessions · {population.contributor_count} "
            f"eligible players · random slope {_text(population.random_effect_slope)} "
            f"{_unit(request.metric_column)}/session · improvement probability "
            f"{_text(population.improvement_probability)}."
        )
        self._refresh_enabled()
        return result

    def _plot(self, result: LongitudinalResult) -> None:
        self.axes.clear()
        frame = pd.DataFrame(asdict(point) for point in result.session_points)
        for player, rows in frame.groupby("player_id", sort=True):
            self.axes.plot(
                rows["session_order"], rows["mean"], marker="o", label=str(player)
            )
        self.axes.set_xlabel("Explicit session order")
        self.axes.set_ylabel(
            f"{result.request.metric_column} ({_unit(result.request.metric_column)})"
        )
        self.axes.set_title("Equal-weight player session means")
        if frame["player_id"].nunique() <= 12:
            self.axes.legend(fontsize=7)
        self.canvas.draw()

    def _table(self, result: LongitudinalResult) -> None:
        columns = (
            "player_id",
            "session_count",
            "slope_per_session",
            "ci_lower",
            "ci_upper",
            "first_to_last_change",
            "status",
        )
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(
            (
                "Player",
                "Sessions",
                "Slope",
                "CI lower",
                "CI upper",
                "First-to-last",
                "Status",
            )
        )
        self.table.setRowCount(len(result.players))
        for row_index, player in enumerate(result.players):
            values = asdict(player)
            for column_index, column in enumerate(columns):
                self.table.setItem(
                    row_index, column_index, QTableWidgetItem(_text(values[column]))
                )
        self.table.resizeColumnsToContents()

    def calculate_safely(self) -> None:
        try:
            self.calculate()
        except ValueError as error:
            QMessageBox.warning(self, "Longitudinal Analysis Unavailable", str(error))

    def document(self) -> dict[str, object]:
        return {} if self.result is None else asdict(self.result)

    def save_plot_dialog(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Save Longitudinal Plot",
            "launch-monitor-longitudinal.svg",
            "SVG (*.svg);;PNG (*.png);;PDF (*.pdf)",
        )
        if selected:
            self.canvas.figure.savefig(Path(selected), dpi=180)


__all__ = ["LaunchMonitorLongitudinalWidget"]
