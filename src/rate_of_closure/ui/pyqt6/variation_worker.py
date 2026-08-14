"""QThread worker running variation studies off the UI thread (#4120 V3).

Same worker shape as :class:`~rate_of_closure.ui.pyqt6.solver_worker.
SolverWorker` (itself modeled on the movement_optimizer plumbing): the
Monte-Carlo batch runs inside this :class:`QThread`, solver-shaped
progress reports cross back as queued signals, and cancellation is
cooperative via the engine's ``cancel_event`` seam.

Optionally follows the main batch with the one-at-a-time sensitivity
pass (``len(plan.noise)`` extra sub-studies, same seed streams).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import replace

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from rate_of_closure.application.workspace_variation_session import (
    VariationAnalysisExecution,
)
from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.variation import (
    build_simulation_ensemble_request,
    run_simulation_ensemble,
)
from shared.python.swing_sim.variation import (
    CancelledError,
    SensitivityResult,
    VariationPlan,
    one_at_a_time_sensitivity,
    run_variation,
)

logger = logging.getLogger(__name__)

__all__ = ["VariationWorker"]


class VariationWorker(QThread):
    """One variation study: construct, ``start()``, listen, ``cancel()``.

    Signals (all delivered on the GUI thread via queued connections):
        progressed(object): A solver-shaped ``ProgressReport`` snapshot
            (``iteration`` = completed runs of the main batch).
        phaseChanged(str): Human-readable phase ("Running…" /
            "Sensitivity…").
        succeeded(object, object): The final ``VariationDataset`` and the
            ``SensitivityResult`` (or ``None`` when not requested).
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
        analysis_execution: VariationAnalysisExecution = (
            VariationAnalysisExecution.BOTH
        ),
        n_workers: int = 4,
        base_simulation_config: SimulationConfig | None = None,
    ) -> None:
        super().__init__()
        self._plan = plan
        if not isinstance(analysis_execution, VariationAnalysisExecution):
            raise TypeError("analysis_execution must be a VariationAnalysisExecution")
        self._analysis_execution = analysis_execution
        self._n_workers = int(n_workers)
        self._cancel_event = threading.Event()
        self._base_simulation_config = base_simulation_config

    @property
    def cancel_event(self) -> threading.Event:
        """The cooperative cancellation event wired into the engine."""
        return self._cancel_event

    @property
    def total_runs(self) -> int:
        """Main-batch run count (drives the determinate progress bar)."""
        return int(self._plan.n_runs)

    def cancel(self) -> None:
        """Request cooperative cancellation (in-flight runs unwind)."""
        self._cancel_event.set()

    def run(self) -> None:  # pragma: no cover — exercised via signals in tests
        """Thread body: run the study, translate outcomes into signals."""
        try:
            if self._cancel_event.is_set():
                raise CancelledError
            run_together = self._analysis_execution in (
                VariationAnalysisExecution.ALL_TOGETHER,
                VariationAnalysisExecution.BOTH,
            )
            run_individually = self._analysis_execution in (
                VariationAnalysisExecution.INDIVIDUAL,
                VariationAnalysisExecution.BOTH,
            )
            ensemble = None
            dataset = None
            if run_together:
                self.phaseChanged.emit("Running…")
                if self._plan.mode == "swing":
                    if self._base_simulation_config is None:
                        raise ValueError(
                            "swing trace studies require a simulation config"
                        )
                    request = build_simulation_ensemble_request(
                        self._plan, self._base_simulation_config
                    )
                    ensemble = run_simulation_ensemble(
                        request,
                        progress_cb=self.progressed.emit,
                        cancel_event=self._cancel_event,
                    )
                    dataset = ensemble.variation
                else:
                    dataset = run_variation(
                        self._plan,
                        n_workers=self._n_workers,
                        progress_cb=self.progressed.emit,
                        cancel_event=self._cancel_event,
                    )
            sensitivity: SensitivityResult | None = None
            if run_individually:
                self.phaseChanged.emit("Sensitivity…")
                sensitivity = (
                    self._simulation_sensitivity()
                    if self._plan.mode == "swing"
                    else one_at_a_time_sensitivity(
                        self._plan,
                        n_workers=self._n_workers,
                        cancel_event=self._cancel_event,
                    )
                )
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 — surface engine failures
            logger.warning("variation run failed: %s", exc)
            self.failed.emit(str(exc))
        else:
            if ensemble is not None:
                self.ensembleSucceeded.emit(ensemble)
            self.succeeded.emit(dataset, sensitivity)

    def _simulation_sensitivity(self) -> SensitivityResult:
        """Run trace-capable one-at-a-time studies through the same simulator."""
        assert self._base_simulation_config is not None
        rows: list[np.ndarray] = []
        output_names: tuple[str, ...] | None = None
        for spec in self._plan.noise:
            if self._cancel_event.is_set():
                raise CancelledError
            sub_plan = replace(self._plan, noise=(spec,), groups=())
            request = build_simulation_ensemble_request(
                sub_plan, self._base_simulation_config
            )
            dataset = run_simulation_ensemble(
                request, cancel_event=self._cancel_event
            ).variation
            current_output_names = dataset.output_names
            output_names = current_output_names
            row = np.full(len(current_output_names), np.nan)
            for column in range(len(current_output_names)):
                values = dataset.outputs[:, column]
                finite = values[np.isfinite(values)]
                if finite.size >= 2:
                    row[column] = float(np.std(finite, ddof=1))
            rows.append(row)
        assert output_names is not None
        matrix = np.vstack(rows)
        with np.errstate(invalid="ignore"):
            column_max = np.nanmax(np.abs(matrix), axis=0)
            denominator = np.where(column_max > 0.0, column_max, 1.0)
            normalized = np.abs(matrix) / denominator
        return SensitivityResult(
            input_keys=tuple(spec.variable_key for spec in self._plan.noise),
            output_names=output_names,
            matrix=matrix,
            normalized=normalized,
        )
