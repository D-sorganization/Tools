"""Worker lifecycle and result-state authority for the PyQt Variation tab."""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QCheckBox, QLabel, QProgressBar, QPushButton

from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.ui.pyqt6.variation_tab_results import populate_result_views
from rate_of_closure.ui.pyqt6.variation_worker import VariationWorker
from rate_of_closure.variation.plot_data import build_ensemble_plot_dataset
from rate_of_closure.variation.simulation_types import SimulationEnsembleResult
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    SensitivityResult,
    VariationDataset,
    VariationPlan,
)

__all__ = ["VariationTabRunMixin"]


class VariationTabRunMixin:
    """Own one generation-safe worker and its scalar/ensemble result state."""

    _simulation_config_valid: bool
    _worker: VariationWorker | None
    _generation: int
    _dataset: VariationDataset | None
    _sensitivity: SensitivityResult | None
    _ensemble_result: SimulationEnsembleResult | None
    _base_simulation_config: SimulationConfig
    _sens_check: QCheckBox
    _run_button: QPushButton
    _cancel_button: QPushButton
    _export_csv: QPushButton
    _export_json: QPushButton
    _export_trace_csv: QPushButton
    _export_ensemble_json: QPushButton
    _progress: QProgressBar
    _status: QLabel
    _ensemble_scatter: Any
    _distribution_matrix: Any
    _arc_overlay: Any
    _landing: Any
    _summary_table: Any
    _sensitivity_table: Any
    _spearman_table: Any
    studyCompleted: Any  # Qt signal descriptor supplied by the concrete tab.

    def build_plan(self) -> VariationPlan:
        """Return the concrete tab's validated plan."""
        raise NotImplementedError

    def _on_run(self) -> None:
        if not self._simulation_config_valid:
            self._status.setText(
                "Cannot run: current Simulation inputs are incomplete or invalid."
            )
            return
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            plan = self.build_plan()
        except (ContractViolationError, ValueError) as exc:
            self._status.setText(f"Cannot run: {exc}")
            return
        self._dataset = None
        self._sensitivity = None
        self._ensemble_result = None
        self._generation += 1
        generation = self._generation
        self._set_running(True)
        worker = VariationWorker(
            plan,
            compute_sensitivity=self._sens_check.isChecked(),
            base_simulation_config=self._base_simulation_config,
        )
        worker.progressed.connect(
            lambda report, current=generation: self._accept_progress(current, report)
        )
        worker.phaseChanged.connect(
            lambda phase, current=generation: self._accept_phase(current, phase)
        )
        worker.succeeded.connect(
            lambda dataset, sensitivity, current=generation: self._accept_succeeded(
                current, dataset, sensitivity
            )
        )
        worker.ensembleSucceeded.connect(
            lambda result, current=generation: self._accept_ensemble_succeeded(
                current, result
            )
        )
        worker.cancelled.connect(
            lambda current=generation: self._accept_cancelled(current)
        )
        worker.failed.connect(
            lambda message, current=generation: self._accept_failed(current, message)
        )
        worker.finished.connect(
            lambda current=generation, owner=worker: self._accept_finished(
                current, owner
            )
        )
        self._worker = worker
        self._progress.setRange(0, worker.total_runs)
        self._progress.setValue(0)
        self._status.setText("Running…")
        worker.start()

    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running and self._simulation_config_valid)
        self._cancel_button.setEnabled(running)
        self._export_csv.setEnabled(not running and self._dataset is not None)
        self._export_json.setEnabled(not running and self._dataset is not None)
        has_ensemble = self._ensemble_result is not None
        self._export_trace_csv.setEnabled(not running and has_ensemble)
        self._export_ensemble_json.setEnabled(not running and has_ensemble)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling…")

    def _is_current_generation(self, generation: int) -> bool:
        return generation == self._generation

    def _accept_progress(self, generation: int, report: object) -> None:
        if self._is_current_generation(generation):
            self._on_progress(report)

    def _accept_phase(self, generation: int, phase: str) -> None:
        if self._is_current_generation(generation):
            self._on_phase(phase)

    def _accept_succeeded(
        self, generation: int, dataset: object, sensitivity: object
    ) -> None:
        if self._is_current_generation(generation) and isinstance(
            dataset, VariationDataset
        ):
            self._on_succeeded(dataset, sensitivity)

    def _accept_ensemble_succeeded(self, generation: int, result: object) -> None:
        if self._is_current_generation(generation) and isinstance(
            result, SimulationEnsembleResult
        ):
            self._on_ensemble_succeeded(result)

    def _accept_cancelled(self, generation: int) -> None:
        if self._is_current_generation(generation):
            self._on_cancelled()

    def _accept_failed(self, generation: int, message: str) -> None:
        if self._is_current_generation(generation):
            self._on_failed(message)

    def _accept_finished(self, generation: int, worker: VariationWorker) -> None:
        owns_current_slot = worker is self._worker
        if owns_current_slot:
            self._worker = None
        if self._is_current_generation(generation):
            self._on_finished()
        elif owns_current_slot:
            self._set_running(False)

    def _on_progress(self, report: object) -> None:
        iteration = int(getattr(report, "iteration", 0))
        failed = int(getattr(report, "cost", 0.0))
        self._progress.setValue(min(iteration, self._progress.maximum()))
        note = f", {failed} failed" if failed else ""
        self._status.setText(
            f"Run {iteration}/{self._progress.maximum()}{note} — "
            f"{getattr(report, 'elapsed_s', 0.0):.1f} s"
        )

    def _on_phase(self, phase: str) -> None:
        if phase.startswith("Sensitivity"):
            self._progress.setRange(0, 0)
        self._status.setText(phase)

    def _on_succeeded(self, dataset: VariationDataset, sensitivity: object) -> None:
        self._dataset = dataset
        self._sensitivity = (
            sensitivity if isinstance(sensitivity, SensitivityResult) else None
        )
        if self._ensemble_result is None:
            self._ensemble_scatter.set_variation_dataset(dataset)
            self._distribution_matrix.set_variation_dataset(dataset)
        self._populate_results()
        failures = dataset.plan.n_runs - dataset.n_success
        note = f" ({failures} runs failed)" if failures else ""
        self._status.setText(
            f"Done: {dataset.n_success}/{dataset.plan.n_runs} runs in "
            f"{dataset.elapsed_s:.1f} s{note}."
        )
        self.studyCompleted.emit(dataset)

    def _on_ensemble_succeeded(self, result: SimulationEnsembleResult) -> None:
        """Populate complete-trace views before the scalar completion callback."""
        self._ensemble_result = result
        self._landing.set_outcomes(tuple(outcome.status for outcome in result.outcomes))
        self._export_trace_csv.setEnabled(True)
        self._export_ensemble_json.setEnabled(True)
        plot_dataset = build_ensemble_plot_dataset(result)
        self._ensemble_scatter.set_plot_dataset(plot_dataset)
        self._distribution_matrix.set_plot_dataset(plot_dataset)
        self._arc_overlay.set_plot_dataset(plot_dataset)

    def _on_cancelled(self) -> None:
        self._status.setText("Cancelled.")

    def _on_failed(self, message: str) -> None:
        self._status.setText(f"Study failed: {message}")

    def _on_finished(self) -> None:
        self._progress.setRange(0, max(self._progress.maximum(), 1))
        self._progress.setValue(self._progress.maximum())
        self._set_running(False)

    def _populate_results(self) -> None:
        dataset = self._dataset
        if dataset is None:
            return
        populate_result_views(
            dataset,
            self._sensitivity,
            self._summary_table,
            self._sensitivity_table,
            self._spearman_table,
            self._landing,
        )

    def _invalidate_current_study(self) -> None:
        self._generation += 1
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        self._dataset = None
        self._sensitivity = None
        self._ensemble_result = None
        self._summary_table.setRowCount(0)
        for table in (self._sensitivity_table, self._spearman_table):
            table.setRowCount(0)
            table.setColumnCount(0)
        self._landing.clear_view()
        self._ensemble_scatter.clear_view()
        self._distribution_matrix.clear_view()
        self._arc_overlay.clear_view()
