"""Generation- and worker-bound lifecycle for the PyQt Variation tab."""

from PyQt6.QtWidgets import QComboBox, QLabel, QProgressBar, QPushButton

from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.ui.pyqt6.variation_tab_results import (
    PreparedResultViews,
    prepare_result_views,
    prepare_sensitivity_view,
)
from rate_of_closure.ui.pyqt6.variation_worker import (
    MAX_WORKER_ERROR_LENGTH,
    VariationWorker,
)
from rate_of_closure.ui.pyqt6.visual_state_frame import VisualStateFrame
from rate_of_closure.variation.analysis_policy import (
    AnalysisExecution,
    runs_individual_analysis,
    runs_joint_analysis,
    validate_analysis_execution,
)
from rate_of_closure.variation.plot_data import (
    EnsemblePlotDataset,
    build_ensemble_plot_dataset,
    scalar_plot_variables,
)
from rate_of_closure.variation.simulation_types import SimulationEnsembleResult
from rate_of_closure.variation_visual_state import (
    VariationVisualEvent,
    simulation_authority_identity,
    variation_visual_state,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import (
    SensitivityResult,
    VariationDataset,
    VariationPlan,
)


class VariationTabRunMixin:
    _simulation_config_valid: bool
    _worker: VariationWorker | None
    _generation: int
    _dataset: VariationDataset | None
    _sensitivity: SensitivityResult | None
    _ensemble_result: SimulationEnsembleResult | None
    _pending_ensemble_result: SimulationEnsembleResult | None
    _accepted_authority_identity: object | None
    _active_authority_identity: object | None
    _active_plan: VariationPlan | None
    _active_analysis_execution: AnalysisExecution
    _accepted_result_views: PreparedResultViews | None
    _accepted_plot_dataset: EnsemblePlotDataset | None
    _base_simulation_config: SimulationConfig
    _analysis_combo: QComboBox
    _run_button: QPushButton
    _cancel_button: QPushButton
    _export_csv: QPushButton
    _export_json: QPushButton
    _export_trace_csv: QPushButton
    _export_ensemble_json: QPushButton
    _progress: QProgressBar
    _status: QLabel
    _visual_frame: VisualStateFrame
    studyCompleted: object

    def build_plan(self) -> VariationPlan:
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
        authority_identity = self._current_authority_identity(plan)
        retains_accepted = authority_identity == self._accepted_authority_identity
        if not retains_accepted:
            self._clear_accepted_result()
            self._accepted_authority_identity = None
        self._pending_ensemble_result = None
        self._active_authority_identity = authority_identity
        self._active_plan = plan
        policy = self._analysis_execution()
        self._active_analysis_execution = policy
        self._generation += 1
        generation = self._generation
        self._set_running(True)
        worker = VariationWorker(
            plan,
            analysis_execution=policy,
            base_simulation_config=self._base_simulation_config,
        )
        worker.progressed.connect(
            lambda report, current=generation, owner=worker: self._accept_progress(
                current, owner, report
            )
        )
        worker.phaseChanged.connect(
            lambda phase, current=generation, owner=worker: self._accept_phase(
                current, owner, phase
            )
        )
        worker.succeeded.connect(
            lambda dataset, sensitivity, current=generation, owner=worker: (
                self._accept_succeeded(current, owner, dataset, sensitivity)
            )
        )
        worker.ensembleSucceeded.connect(
            lambda result, current=generation, owner=worker: (
                self._accept_ensemble_succeeded(current, owner, result)
            )
        )
        worker.cancelled.connect(
            lambda current=generation, owner=worker: self._accept_cancelled(
                current, owner
            )
        )
        worker.failed.connect(
            lambda message, current=generation, owner=worker: self._accept_failed(
                current, owner, message
            )
        )
        worker.finished.connect(lambda owner=worker: self._accept_finished(owner))
        self._worker = worker
        self._progress.setRange(0, worker.total_runs)
        self._progress.setValue(0)
        self._status.setText("Running…")
        self._set_visual_event(
            VariationVisualEvent.START_RETAINED
            if retains_accepted
            else VariationVisualEvent.START_EMPTY,
            "Running…",
        )
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
            self._generation += 1
            self._worker.cancel()
            self._pending_ensemble_result = None
            self._status.setText("Cancelled: no partial variation result was accepted.")
            self._set_visual_event(
                VariationVisualEvent.CANCEL_RETAINED
                if self._retains_active_result()
                else VariationVisualEvent.CANCEL_EMPTY,
                "Cancelled: no partial variation result was accepted.",
            )
            self._clear_active_authority()

    def _accepts_worker_event(self, generation: int, owner: VariationWorker) -> bool:
        if generation != self._generation or owner is not self._worker:
            return False
        active = self._active_authority_identity
        if active is None:
            return False
        if owner.authority_identity != active:
            owner.cancel()
            self._on_failed("Internal worker authority mismatch; result rejected.")
            return False
        return True

    def _accept_progress(
        self, generation: int, owner: VariationWorker, report: object
    ) -> None:
        if self._accepts_worker_event(generation, owner):
            self._on_progress(report)

    def _accept_phase(
        self, generation: int, owner: VariationWorker, phase: str
    ) -> None:
        if self._accepts_worker_event(generation, owner):
            self._on_phase(phase)

    def _accept_succeeded(
        self,
        generation: int,
        owner: VariationWorker,
        dataset: object,
        sensitivity: object,
    ) -> None:
        if not self._accepts_worker_event(generation, owner):
            return
        expects_dataset = runs_joint_analysis(self._active_analysis_execution)
        if expects_dataset != isinstance(dataset, VariationDataset):
            owner.cancel()
            self._on_failed(
                "Internal worker returned unexpected joint-analysis availability."
            )
            return
        accepted_dataset = dataset if isinstance(dataset, VariationDataset) else None
        self._on_succeeded(accepted_dataset, sensitivity)

    def _accept_ensemble_succeeded(
        self, generation: int, owner: VariationWorker, result: object
    ) -> None:
        if not self._accepts_worker_event(generation, owner):
            return
        if not isinstance(result, SimulationEnsembleResult):
            owner.cancel()
            self._on_failed("Internal worker returned an invalid swing ensemble.")
            return
        self._on_ensemble_succeeded(result)

    def _accept_cancelled(self, generation: int, owner: VariationWorker) -> None:
        if self._accepts_worker_event(generation, owner):
            self._on_cancelled()

    def _accept_failed(
        self, generation: int, owner: VariationWorker, message: str
    ) -> None:
        if self._accepts_worker_event(generation, owner):
            self._on_failed(message)

    def _accept_finished(self, worker: VariationWorker) -> None:
        if worker is not self._worker:
            return
        self._worker = None
        self._on_finished()

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
        self._status.setText(phase)

    def _on_succeeded(
        self, dataset: VariationDataset | None, sensitivity: object
    ) -> None:
        active_plan = self._active_plan
        if self._active_authority_identity is None or active_plan is None:
            return
        if dataset is not None and dataset.plan != active_plan:
            self._on_failed("Result plan does not match the active variation request.")
            return
        expects_dataset = runs_joint_analysis(self._active_analysis_execution)
        expects_sensitivity = runs_individual_analysis(self._active_analysis_execution)
        if expects_dataset != (dataset is not None):
            self._on_failed(
                "Joint-analysis result availability does not match request."
            )
            return
        if expects_sensitivity != isinstance(sensitivity, SensitivityResult):
            self._on_failed(
                "Sensitivity result availability does not match the active request."
            )
            return
        pending_ensemble = self._pending_ensemble_result
        if active_plan.mode == "swing" and expects_dataset:
            if (
                pending_ensemble is None
                or pending_ensemble.variation.plan != active_plan
                or dataset is not pending_ensemble.variation
            ):
                self._on_failed(
                    "Swing ensemble does not match the active variation request."
                )
                return
        elif pending_ensemble is not None:
            self._on_failed("Unexpected swing ensemble for a scalar variation request.")
            return
        candidate_sensitivity = (
            sensitivity if isinstance(sensitivity, SensitivityResult) else None
        )
        try:
            if dataset is not None:
                prepared_views = prepare_result_views(dataset, candidate_sensitivity)
            else:
                assert candidate_sensitivity is not None
                prepared_views = prepare_sensitivity_view(
                    active_plan, candidate_sensitivity
                )
            plot_dataset = (
                build_ensemble_plot_dataset(pending_ensemble)
                if pending_ensemble is not None
                else None
            )
            if dataset is not None and pending_ensemble is None:
                scalar_plot_variables(dataset)
        except Exception as exc:
            self._on_failed(f"Could not prepare accepted result visuals: {exc}")
            return
        previous = (
            self._dataset,
            self._ensemble_result,
            self._accepted_result_views,
            self._accepted_plot_dataset,
        )
        try:
            self._apply_prepared_result(  # type: ignore[attr-defined]
                prepared_views, pending_ensemble, plot_dataset
            )
        except Exception as exc:
            self._restore_prepared_result(previous)  # type: ignore[attr-defined]
            self._on_failed(f"Could not publish accepted result visuals: {exc}")
            return
        self._dataset = dataset
        self._sensitivity = candidate_sensitivity
        self._ensemble_result = pending_ensemble
        self._accepted_result_views = prepared_views
        self._accepted_plot_dataset = plot_dataset
        self._pending_ensemble_result = None
        if dataset is None:
            self._status.setText(
                "Done: individual one-at-a-time sensitivity completed without "
                "a joint dataset."
            )
        else:
            failures = dataset.plan.n_runs - dataset.n_success
            note = f" ({failures} runs failed)" if failures else ""
            self._status.setText(
                f"Done: {dataset.n_success}/{dataset.plan.n_runs} runs in "
                f"{dataset.elapsed_s:.1f} s{note}."
            )
        self._accepted_authority_identity = self._active_authority_identity
        self._clear_active_authority()
        self._set_visual_event(VariationVisualEvent.SUCCEED, self._status.text())
        if dataset is not None:
            self.studyCompleted.emit(dataset)  # type: ignore[attr-defined]

    def _on_ensemble_succeeded(self, result: SimulationEnsembleResult) -> None:
        self._pending_ensemble_result = result

    def _on_cancelled(self) -> None:
        if self._active_authority_identity is None:
            return
        self._pending_ensemble_result = None
        self._status.setText("Cancelled.")
        self._set_visual_event(
            VariationVisualEvent.CANCEL_RETAINED
            if self._retains_active_result()
            else VariationVisualEvent.CANCEL_EMPTY,
            self._status.text(),
        )
        self._clear_active_authority()

    def _on_failed(self, message: str) -> None:
        if self._active_authority_identity is None:
            return
        self._pending_ensemble_result = None
        bounded = str(message)[:MAX_WORKER_ERROR_LENGTH]
        self._status.setText(f"Study failed: {bounded}")
        self._set_visual_event(
            VariationVisualEvent.FAIL_RETAINED
            if self._retains_active_result()
            else VariationVisualEvent.FAIL_EMPTY,
            self._status.text(),
        )
        self._clear_active_authority()

    def _on_finished(self) -> None:
        self._progress.setRange(0, max(self._progress.maximum(), 1))
        self._progress.setValue(self._progress.maximum())
        self._set_running(False)

    def _invalidate_current_study(self) -> None:
        self._generation += 1
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        self._clear_accepted_result()
        self._clear_active_authority()
        self._accepted_authority_identity = None
        self._set_running(False)
        self._set_visual_event(
            VariationVisualEvent.INVALIDATE,
            "Ready: configuration changed; run again.",
        )

    def _set_visual_event(self, event: VariationVisualEvent, text: str) -> None:
        self._visual_frame.set_state(variation_visual_state(event), text)

    def _retains_active_result(self) -> bool:
        return (
            self._accepted_authority_identity is not None
            and self._accepted_authority_identity == self._active_authority_identity
        )

    def _current_authority_identity(self, plan: VariationPlan) -> object:
        return simulation_authority_identity(
            plan, self._base_simulation_config, self._analysis_execution()
        )

    def _analysis_execution(self) -> AnalysisExecution:
        return validate_analysis_execution(self._analysis_combo.currentData())

    def _clear_active_authority(self) -> None:
        self._active_authority_identity = None
        self._active_plan = None
        self._active_analysis_execution = "both"

    def _clear_accepted_result(self) -> None:
        self._dataset = None
        self._sensitivity = None
        self._ensemble_result = None
        self._accepted_result_views = None
        self._accepted_plot_dataset = None
        self._pending_ensemble_result = None
        self._clear_result_widgets()  # type: ignore[attr-defined]
