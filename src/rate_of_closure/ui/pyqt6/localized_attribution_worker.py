"""Dedicated worker for an explicitly requested localized paired study."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import replace

from PyQt6.QtCore import QThread, pyqtSignal

from rate_of_closure.simulation import run_simulation
from rate_of_closure.variation.localized_attribution_producer import (
    LocalizedAttributionDesign,
    LocalizedAttributionProduction,
    produce_localized_attribution,
)
from rate_of_closure.variation.trial_projection import SimulationExecutor
from shared.python.swing_sim.solver.solve import CancelledError, ProgressReport

logger = logging.getLogger(__name__)
AttributionProducer = Callable[..., LocalizedAttributionProduction]


class LocalizedAttributionWorker(QThread):
    """Execute one explicit baseline/one-source paired study off the UI thread."""

    progressed = pyqtSignal(object)  # noqa: N815 - Qt naming convention
    succeeded = pyqtSignal(object)  # noqa: N815
    cancelled = pyqtSignal()  # noqa: N815
    failed = pyqtSignal(str)  # noqa: N815

    def __init__(
        self,
        design: LocalizedAttributionDesign,
        *,
        executor: SimulationExecutor = run_simulation,
        producer: AttributionProducer = produce_localized_attribution,
    ) -> None:
        super().__init__()
        if not isinstance(design, LocalizedAttributionDesign):
            raise TypeError("design must be LocalizedAttributionDesign")
        self._design = design
        self._executor = executor
        self._producer = producer
        self._cancel_event = threading.Event()
        self._completed = 0

    @property
    def total_runs(self) -> int:
        """Return the exact baseline/perturbed trial count."""
        return 2 * len(self._design.source_plan.noise)

    @property
    def cancel_event(self) -> threading.Event:
        """Expose the cooperative cancellation authority for tests and shutdown."""
        return self._cancel_event

    def cancel(self) -> None:
        """Request cooperative cancellation without discarding prior authority."""
        self._cancel_event.set()

    def run(self) -> None:  # pragma: no cover - exercised through Qt signals
        """Translate producer completion, cancellation, and failure into Qt signals."""
        try:
            if self._cancel_event.is_set():
                raise CancelledError
            production = self._producer(
                self._design,
                executor=self._executor,
                progress_cb=self._report_progress,
                cancel_event=self._cancel_event,
            )
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - worker boundary surfaces failures
            logger.warning("localized paired study failed: %s", exc)
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(production)

    def _report_progress(self, report: ProgressReport) -> None:
        """Expand chunk completion into exact monotonically completed trials."""
        completed = min(int(report.iteration), self.total_runs)
        for iteration in range(self._completed + 1, completed + 1):
            self.progressed.emit(replace(report, iteration=iteration))
        self._completed = max(self._completed, completed)


__all__ = ["LocalizedAttributionWorker"]
