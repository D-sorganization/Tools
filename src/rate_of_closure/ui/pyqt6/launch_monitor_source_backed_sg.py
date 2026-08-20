"""PyQt controls for hash-verified, course-state strokes gained."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)

from rate_of_closure.launch_monitor_strokes_gained import (
    SourceBackedStrokesGainedRequest,
    SourceBackedStrokesGainedResult,
    StrokesGainedBaseline,
    calculate_source_backed_strokes_gained,
    load_strokes_gained_baseline,
)


class LaunchMonitorSourceBackedStrokesGainedWidget(QWidget):
    """Load a verified baseline and map explicit before/after course state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._baseline: StrokesGainedBaseline | None = None
        self._baseline_path: Path | None = None
        self.result: SourceBackedStrokesGainedResult | None = None
        self._build_ui()

    def _combo(self, name: str) -> QComboBox:
        combo = QComboBox()
        combo.setAccessibleName(name)
        combo.setToolTip(name)
        return combo

    def _build_ui(self) -> None:
        self.load_button = QPushButton("Load Verified Baseline...")
        self.before_lie = self._combo("Before lie column")
        self.before_distance = self._combo("Before distance column")
        self.before_unit = self._combo("Before distance unit")
        self.after_lie = self._combo("After lie column")
        self.after_distance = self._combo("After distance column")
        self.after_unit = self._combo("After distance unit")
        self.before_unit.addItems(["yd", "m"])
        self.after_unit.addItems(["yd", "m"])
        self.calculate_button = QPushButton("Calculate Source-Backed SG")
        self.export_button = QPushButton("Export Source-Backed SG...")
        self.status = QLabel(
            "Unavailable until a versioned, hash-verified expected-strokes "
            "baseline artifact is loaded. No table is bundled."
        )
        self.status.setWordWrap(True)
        descriptions = (
            (self.load_button, "Load and SHA-256 verify a licensed baseline artifact"),
            (
                self.calculate_button,
                "Calculate verified E(before) minus one minus verified E(after)",
            ),
            (
                self.export_button,
                "Export baseline provenance, formula, lookups, and shot results",
            ),
        )
        for control, description in descriptions:
            control.setAccessibleName(description)
            control.setToolTip(description)
        self._install_layout()
        self.load_button.clicked.connect(self.load_dialog)
        self.calculate_button.clicked.connect(self.calculate_safely)
        self.export_button.clicked.connect(self.export_dialog)
        for combo in (
            self.before_lie,
            self.before_distance,
            self.after_lie,
            self.after_distance,
        ):
            combo.currentTextChanged.connect(self._refresh_enabled)
        self._refresh_enabled()

    def _install_layout(self) -> None:
        form = QFormLayout(self)
        form.addRow(self.load_button)
        form.addRow("Before lie:", self.before_lie)
        form.addRow("Before distance:", self.before_distance)
        form.addRow("Before unit:", self.before_unit)
        form.addRow("After lie:", self.after_lie)
        form.addRow("After distance:", self.after_distance)
        form.addRow("After unit:", self.after_unit)
        form.addRow(self.calculate_button)
        form.addRow(self.export_button)
        form.addRow(self.status)

    def set_dataset(self, frame: pd.DataFrame) -> None:
        """Bind retained rows while retaining an independently verified baseline."""

        self._frame = frame
        columns = sorted(str(column) for column in frame.columns)
        numeric = [
            str(column)
            for column in frame.columns
            if pd.to_numeric(frame[column], errors="coerce").notna().any()
        ]
        for combo, values in (
            (self.before_lie, columns),
            (self.after_lie, columns),
            (self.before_distance, numeric),
            (self.after_distance, numeric),
        ):
            combo.clear()
            combo.addItem("")
            combo.addItems(values)
        self.result = None
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        ready = self._baseline is not None and all(
            combo.currentText()
            for combo in (
                self.before_lie,
                self.before_distance,
                self.after_lie,
                self.after_distance,
            )
        )
        self.calculate_button.setEnabled(ready)
        self.export_button.setEnabled(self.result is not None)

    def load_path(self, path: Path) -> StrokesGainedBaseline:
        """Load, validate, and display one baseline artifact."""

        baseline = load_strokes_gained_baseline(path)
        self._baseline = baseline
        self._baseline_path = path
        self.result = None
        self.status.setText(
            f"Verified {baseline.baseline_id} · version {baseline.version} · "
            f"SHA-256 {baseline.table_sha256} · license {baseline.license}."
        )
        self._refresh_enabled()
        return baseline

    def load_dialog(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Load Strokes-Gained Baseline", "", "JSON (*.json)"
        )
        if selected:
            try:
                self.load_path(Path(selected))
            except (OSError, ValueError, json.JSONDecodeError) as error:
                QMessageBox.warning(self, "Baseline Not Loaded", str(error))

    def calculate(self) -> SourceBackedStrokesGainedResult:
        if self._baseline is None:
            raise ValueError("a verified strokes-gained baseline is required")
        result = calculate_source_backed_strokes_gained(
            self._frame,
            self._baseline,
            SourceBackedStrokesGainedRequest(
                self.before_lie.currentText(),
                self.before_distance.currentText(),
                self.after_lie.currentText(),
                self.after_distance.currentText(),
                self.before_unit.currentText(),
                self.after_unit.currentText(),
            ),
        )
        self.result = result
        self.status.setText(
            f"Mean source-backed SG {result.mean:.3f} strokes across "
            f"{len(result.values)} shots · {result.baseline_id} "
            f"{result.baseline_version}."
        )
        self._refresh_enabled()
        return result

    def calculate_safely(self) -> None:
        try:
            self.calculate()
        except ValueError as error:
            QMessageBox.warning(self, "Source-Backed SG Unavailable", str(error))

    def document(self) -> dict[str, object] | None:
        if self.result is None:
            return None
        return {
            **asdict(self.result),
            "baseline_relative_path": str(self._baseline_path or ""),
        }

    def export_dialog(self) -> None:
        document = self.document()
        if document is None:
            return
        selected, _ = QFileDialog.getSaveFileName(
            self, "Export Source-Backed SG", "source-backed-sg.json", "JSON (*.json)"
        )
        if selected:
            Path(selected).write_text(json.dumps(document, indent=2), encoding="utf-8")


__all__ = ["LaunchMonitorSourceBackedStrokesGainedWidget"]
