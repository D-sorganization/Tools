"""QThread worker running variation studies off the UI thread (#4120 V3).

Same worker shape as :class:`~rate_of_closure.ui.pyqt6.solver_worker.
SolverWorker` (itself modeled on the movement_optimizer plumbing): the
Monte-Carlo batch runs inside this :class:`QThread`, solver-shaped
progress reports cross back as queued signals, and cancellation is
cooperative via the engine's ``cancel_event`` seam.

Runs the selected joint batch, one-at-a-time sensitivity batches, or both.
Individual studies use the same seed streams as the joint plan.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.variation import (
    build_simulation_ensemble_request,
    run_simulation_ensemble,
)
from rate_of_closure.variation.analysis_policy import (
    AnalysisExecution,
    planned_analysis_runs,
    runs_individual_analysis,
    runs_joint_analysis,
    validate_analysis_execution,
)
from rate_of_closure.variation_visual_state import simulation_authority_identity
from shared.python.swing_sim.solver.solve import ProgressReport
from shared.python.swing_sim.variation import (
    CancelledError,
    SensitivityResult,
    VariationPlan,
    finite_sample_standard_deviation,
    one_at_a_time_sensitivity,
    run_variation,
    sensitivity_from_standard_deviations,
)

logger = logging.getLogger(__name__)

__all__ = ["MAX_WORKER_ERROR_LENGTH", "VariationWorker"]

MAX_WORKER_ERROR_LENGTH = 512


def _emit_offset_progress(
    callback: Callable[[ProgressReport], None],
    iteration_offset: int,
    failure_offset: int,
    report: ProgressReport,
) -> None:
    """Translate sub-study progress onto the complete analysis axis."""
    callback(
        replace(
            report,
            iteration=iteration_offset + report.iteration,
            cost=failure_offset + report.cost,
        )
    )


class VariationWorker(QThread):
    """One variation study: construct, ``start()``, listen, ``cancel()``.

    Signals (all delivered on the GUI thread via queued connections):
        progressed(object): A solver-shaped ``ProgressReport`` snapshot
            (``iteration`` = completed runs of the main batch).
        phaseChanged(str): Human-readable phase ("Running…" /
            "Sensitivity…").
        succeeded(object, object): The final ``VariationDataset`` and
            ``SensitivityResult``. Either object can be ``None`` when its
            corresponding analysis was not selected.
        cancelled(): The study was cancelled.
        failed(str): The engine raised; message is user-presentable.
    """

    progressed = pyqtSignal(object)  # noqa: N815 — Qt signal convention
    phaseChanged = pyqtSignal(str)  # noqa: N815
    succeeded = pyqtSignal(object, object)  # noqa: N815
    cancelled = pyqtSignal()  # noqa: N815
    failed = pyqtSignal(str)  # noqa: N815
    ensembleSucceeded = pyqtSignal(object)  # noqa: N815

    def __init__(
        self,
        plan: VariationPlan,
        analysis_execution: AnalysisExecution = "both",
        n_workers: int = 4,
        base_simulation_config: SimulationConfig | None = None,
    ) -> None:
        super().__init__()
        self._plan = plan
        self._analysis_execution = validate_analysis_execution(analysis_execution)
        self._n_workers = int(n_workers)
        self._cancel_event = threading.Event()
        self._base_simulation_config = base_simulation_config
        self._authority_identity = (
            simulation_authority_identity(
                self._plan, base_simulation_config, self._analysis_execution
            )
            if base_simulation_config is not None
            else None
        )

    @property
    def authority_identity(self) -> object | None:
        """Exact immutable in-session identity of captured execution inputs."""
        return self._authority_identity

    @property
    def cancel_event(self) -> threading.Event:
        """The cooperative cancellation event wired into the engine."""
        return self._cancel_event

    @property
    def total_runs(self) -> int:
        """Exact joint plus individual evaluation count."""
        return int(
            planned_analysis_runs(
                self._plan.n_runs,
                len(self._plan.noise),
                self._analysis_execution,
            )
        )

    def cancel(self) -> None:
        """Request cooperative cancellation (in-flight runs unwind)."""
        self._cancel_event.set()

    def run(self) -> None:  # pragma: no cover — exercised via signals in tests
        """Thread body: run the study, translate outcomes into signals."""
        try:
            if self._cancel_event.is_set():
                raise CancelledError
            started = time.monotonic()
            ensemble = None
            dataset = None
            if runs_joint_analysis(self._analysis_execution):
                self.phaseChanged.emit("Running jointly enabled variables…")
            if (
                runs_joint_analysis(self._analysis_execution)
                and self._plan.mode == "swing"
            ):
                if self._base_simulation_config is None:
                    raise ValueError("swing trace studies require a simulation config")
                request = build_simulation_ensemble_request(
                    self._plan, self._base_simulation_config
                )
                ensemble = run_simulation_ensemble(
                    request,
                    progress_cb=self.progressed.emit,
                    cancel_event=self._cancel_event,
                )
                dataset = ensemble.variation
            elif runs_joint_analysis(self._analysis_execution):
                dataset = run_variation(
                    self._plan,
                    n_workers=self._n_workers,
                    progress_cb=self.progressed.emit,
                    cancel_event=self._cancel_event,
                )
            sensitivity: SensitivityResult | None = None
            if runs_individual_analysis(self._analysis_execution):
                self.phaseChanged.emit("Sensitivity: individual interventions…")
                completed_before = self._plan.n_runs if dataset is not None else 0
                failed_before = (
                    self._plan.n_runs - dataset.n_success if dataset is not None else 0
                )

                def individual_progress(report: ProgressReport) -> None:
                    self.progressed.emit(
                        replace(
                            report,
                            iteration=completed_before + report.iteration,
                            cost=failed_before + report.cost,
                            elapsed_s=time.monotonic() - started,
                        )
                    )

                sensitivity = (
                    self._simulation_sensitivity(individual_progress)
                    if self._plan.mode == "swing"
                    else one_at_a_time_sensitivity(
                        self._plan,
                        n_workers=self._n_workers,
                        cancel_event=self._cancel_event,
                        progress_cb=individual_progress,
                    )
                )
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 — surface engine failures
            logger.warning("variation run failed: %s", exc)
            self.failed.emit(str(exc)[:MAX_WORKER_ERROR_LENGTH])
        else:
            if ensemble is not None:
                self.ensembleSucceeded.emit(ensemble)
            self.succeeded.emit(dataset, sensitivity)

    def _simulation_sensitivity(
        self, progress_cb: Callable[[ProgressReport], None]
    ) -> SensitivityResult:
        """Run trace-capable one-at-a-time studies through the same simulator."""
        assert self._base_simulation_config is not None
        rows: list[np.ndarray] = []
        output_names: tuple[str, ...] | None = None
        completed_before = 0
        failed_before = 0
        for spec in self._plan.noise:
            if self._cancel_event.is_set():
                raise CancelledError
            sub_plan = replace(self._plan, noise=(spec,), groups=())
            request = build_simulation_ensemble_request(
                sub_plan, self._base_simulation_config
            )
            offset = completed_before
            prior_failures = failed_before

            dataset = run_simulation_ensemble(
                request,
                progress_cb=partial(
                    _emit_offset_progress,
                    progress_cb,
                    offset,
                    prior_failures,
                ),
                cancel_event=self._cancel_event,
            ).variation
            completed_before += self._plan.n_runs
            failed_before += self._plan.n_runs - dataset.n_success
            current_output_names = dataset.output_names
            output_names = current_output_names
            row = np.full(len(current_output_names), np.nan)
            for column in range(len(current_output_names)):
                values = dataset.outputs[:, column]
                finite = values[np.isfinite(values)]
                if finite.size >= 2:
                    row[column] = finite_sample_standard_deviation(finite)
            rows.append(row)
        assert output_names is not None
        return sensitivity_from_standard_deviations(
            tuple(spec.variable_key for spec in self._plan.noise),
            output_names,
            np.vstack(rows),
        )
