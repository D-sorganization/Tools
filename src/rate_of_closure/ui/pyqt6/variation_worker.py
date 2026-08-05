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

from PyQt6.QtCore import QThread, pyqtSignal

from shared.python.swing_sim.variation import (
    CancelledError,
    SensitivityResult,
    VariationDataset,
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

    def __init__(
        self,
        plan: VariationPlan,
        compute_sensitivity: bool = True,
        n_workers: int = 4,
    ) -> None:
        super().__init__()
        self._plan = plan
        self._compute_sensitivity = bool(compute_sensitivity)
        self._n_workers = int(n_workers)
        self._cancel_event = threading.Event()

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
            self.phaseChanged.emit("Running…")
            dataset: VariationDataset = run_variation(
                self._plan,
                n_workers=self._n_workers,
                progress_cb=self.progressed.emit,
                cancel_event=self._cancel_event,
            )
            sensitivity: SensitivityResult | None = None
            if self._compute_sensitivity:
                self.phaseChanged.emit("Sensitivity…")
                sensitivity = one_at_a_time_sensitivity(
                    self._plan,
                    n_workers=self._n_workers,
                    cancel_event=self._cancel_event,
                )
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 — surface engine failures
            logger.warning("variation run failed: %s", exc)
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(dataset, sensitivity)
