"""Plan and dataset persistence behavior for the PyQt Variation tab."""

from __future__ import annotations

import json
from typing import cast

from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from rate_of_closure.ui.pyqt6.variation_rows import NoiseRow
from rate_of_closure.variation import (
    ChipForgivenessStudy,
    chip_forgiveness_study_to_csv,
    chip_forgiveness_study_to_json,
)
from rate_of_closure.variation.ensemble_io import (
    write_json as write_ensemble_json,
)
from rate_of_closure.variation.ensemble_io import (
    write_trace_csv,
)
from rate_of_closure.variation.simulation_types import SimulationEnsembleResult
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import MODES, VariationDataset, VariationPlan
from shared.python.swing_sim.variation.dataset_io import write_csv, write_json


class VariationTabIoMixin:
    """Keep persistence responsibilities separate from study orchestration."""

    _dataset: VariationDataset | None
    _ensemble_result: SimulationEnsembleResult | None
    _forgiveness_study: ChipForgivenessStudy | None
    _status: QLabel
    _mode_combo: QComboBox
    _runs_spin: QSpinBox
    _seed_spin: QSpinBox
    _flight_combo: QComboBox
    _base_combo: QComboBox
    _loaded_base: dict[str, float]
    _rows: list[NoiseRow]

    def build_plan(self) -> VariationPlan:
        """Return the current plan; implemented by the concrete tab."""
        raise NotImplementedError

    def _add_row(self) -> NoiseRow:
        """Add a noise row; implemented by the concrete tab."""
        raise NotImplementedError

    def _remove_row(self, row: NoiseRow) -> None:
        """Remove a noise row; implemented by the concrete tab."""
        raise NotImplementedError

    def _build_export_box(self) -> QGroupBox:
        """Build dataset, trace, and reproducibility exports."""
        box = QGroupBox("Export / Reproduce")
        layout = QHBoxLayout(box)
        self._export_csv = QPushButton("Dataset CSV")
        self._export_csv.setToolTip(
            "Export the runs table (inputs, outputs, success flags) as CSV."
        )
        self._export_csv.clicked.connect(self._on_export_csv)
        self._export_json = QPushButton("Dataset JSON")
        self._export_json.setToolTip(
            "Export the full dataset + plan as JSON (documented, re-importable)."
        )
        self._export_json.clicked.connect(self._on_export_json)
        self._export_trace_csv = QPushButton("Swing Traces CSV")
        self._export_trace_csv.setToolTip(
            "Export every trial, sample, and modeled point in the explicit app frame."
        )
        self._export_trace_csv.clicked.connect(self._on_export_trace_csv)
        self._export_ensemble_json = QPushButton("Swing Ensemble JSON")
        self._export_ensemble_json.setToolTip(
            "Export the complete plan, typed outcomes, scalar rows, "
            "and position traces."
        )
        self._export_ensemble_json.clicked.connect(self._on_export_ensemble_json)
        self._export_forgiveness_csv = QPushButton("Forgiveness CSV")
        self._export_forgiveness_csv.setToolTip(
            "Export every chip trial, cohort, loss, sampled input, "
            "and available metric."
        )
        self._export_forgiveness_csv.clicked.connect(self._on_export_forgiveness_csv)
        self._export_forgiveness_json = QPushButton("Forgiveness JSON")
        self._export_forgiveness_json.setToolTip(
            "Export the qualified chip objective, population, records, confidence, "
            "CVaR, convergence, and limitations."
        )
        self._export_forgiveness_json.clicked.connect(self._on_export_forgiveness_json)
        save_plan = QPushButton("Save Plan")
        save_plan.setToolTip(
            "Save just the plan as JSON — the schema the web tab also reads."
        )
        save_plan.clicked.connect(self._on_save_plan)
        load_plan = QPushButton("Load Plan")
        load_plan.setToolTip("Load a plan JSON back into the editors.")
        load_plan.clicked.connect(self._on_load_plan)
        exports = (
            self._export_csv,
            self._export_json,
            self._export_trace_csv,
            self._export_ensemble_json,
            self._export_forgiveness_csv,
            self._export_forgiveness_json,
        )
        for button in exports:
            button.setEnabled(False)
        for button in (*exports, save_plan, load_plan):
            layout.addWidget(button)
        return box

    def _on_export_csv(self) -> None:
        if self._dataset is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Export Dataset CSV",
            "variation_dataset.csv",
            "CSV (*.csv)",
        )
        if path:
            write_csv(self._dataset, path)
            self._status.setText(f"Dataset CSV written to {path}.")

    def _on_export_json(self) -> None:
        if self._dataset is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Export Dataset JSON",
            "variation_dataset.json",
            "JSON (*.json)",
        )
        if path:
            write_json(self._dataset, path)
            self._status.setText(f"Dataset JSON written to {path}.")

    def _on_export_trace_csv(self) -> None:
        if self._ensemble_result is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Export Swing Traces CSV",
            "variation_swing_traces.csv",
            "CSV (*.csv)",
        )
        if path:
            write_trace_csv(self._ensemble_result, path)
            self._status.setText(f"Long-form swing traces written to {path}.")

    def _on_export_ensemble_json(self) -> None:
        if self._ensemble_result is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Export Complete Swing Ensemble JSON",
            "variation_swing_ensemble.json",
            "JSON (*.json)",
        )
        if path:
            write_ensemble_json(self._ensemble_result, path)
            self._status.setText(f"Complete swing ensemble written to {path}.")

    def _on_export_forgiveness_csv(self) -> None:
        if self._forgiveness_study is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Export Chip Forgiveness Trials CSV",
            "chip_forgiveness_trials.csv",
            "CSV (*.csv)",
        )
        if path:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(chip_forgiveness_study_to_csv(self._forgiveness_study))
            self._status.setText(f"Chip forgiveness trials written to {path}.")

    def _on_export_forgiveness_json(self) -> None:
        if self._forgiveness_study is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Export Chip Forgiveness Study JSON",
            "chip_forgiveness_study.json",
            "JSON (*.json)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(chip_forgiveness_study_to_json(self._forgiveness_study))
            self._status.setText(f"Chip forgiveness study written to {path}.")

    def _on_save_plan(self) -> None:
        try:
            plan = self.build_plan()
        except (ContractViolationError, ValueError) as exc:
            self._status.setText(f"Cannot save plan: {exc}")
            return
        path, _filter = QFileDialog.getSaveFileName(
            cast(QWidget, self),
            "Save Variation Plan",
            "variation_plan.json",
            "JSON (*.json)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(plan.dumps())
            self._status.setText(f"Plan saved to {path}.")

    def _on_load_plan(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            cast(QWidget, self), "Load Variation Plan", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                plan = VariationPlan.loads(handle.read())
        except (ContractViolationError, ValueError, json.JSONDecodeError) as exc:
            self._status.setText(f"Cannot load plan: {exc}")
            return
        self.load_plan(plan)
        self._status.setText(f"Plan loaded from {path}.")

    def load_plan(self, plan: VariationPlan) -> None:
        """Drive all editors from a plan (used by Load Plan and tests)."""
        self.require_plan_loadable(plan)
        self._mode_combo.setCurrentIndex(MODES.index(plan.mode))
        self._runs_spin.setValue(plan.n_runs)
        self._seed_spin.setValue(plan.seed)
        self._flight_combo.setCurrentText(plan.flight_model)
        self._base_combo.setCurrentIndex(0)
        self._loaded_base = dict(plan.base_variables)
        while len(self._rows) > 1:
            self._remove_row(self._rows[-1])
        for index, spec in enumerate(plan.noise):
            row = self._rows[0] if index == 0 else self._add_row()
            row.load_spec(spec)

    def require_plan_loadable(self, plan: VariationPlan) -> None:
        """Fail before mutation when a valid wire plan exceeds editor bounds."""
        if type(plan) is not VariationPlan:
            raise TypeError("plan must be an exact VariationPlan")
        if plan.n_runs > self._runs_spin.maximum():
            raise ValueError(
                f"plan n_runs exceeds the editor maximum {self._runs_spin.maximum()}"
            )


__all__ = ["VariationTabIoMixin"]
