"""Responsive QThread wrapper for capability optimization."""

from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from rate_of_closure.application.capability_workflow import CapabilityWorkflowDocument
from rate_of_closure.variation.capability_observation_adapter import (
    CapabilityObservationEnsembleBuilder,
)
from shared.python.swing_sim.flight.capability_flight_evaluator import (
    make_capability_flight_evaluator,
)
from shared.python.swing_sim.flight.capability_observation import (
    CapabilityOptimizationCancelled,
    CapabilityOptimizationHooks,
    CapabilitySampleObservation,
)
from shared.python.swing_sim.flight.capability_optimizer import optimize_capability

logger = logging.getLogger(__name__)


class CapabilityOptimizationWorker(QThread):
    """Run an immutable capability workflow off the GUI thread."""

    progressed = pyqtSignal(int, int)  # noqa: N815
    succeeded = pyqtSignal(object, object)  # noqa: N815
    cancelled = pyqtSignal(int, int)  # noqa: N815
    failed = pyqtSignal(str)  # noqa: N815

    def __init__(self, document: CapabilityWorkflowDocument) -> None:
        super().__init__()
        if not isinstance(document, CapabilityWorkflowDocument):
            raise TypeError("document must be a CapabilityWorkflowDocument")
        self._document = document
        self._cancel_event = threading.Event()
        self._last_progress = 0

    def cancel(self) -> None:
        """Request cooperative cancellation at the next sample boundary."""
        self._cancel_event.set()

    def _accept(
        self,
        builder: CapabilityObservationEnsembleBuilder,
        observation: CapabilitySampleObservation,
    ) -> None:
        builder.accept(observation)
        completed, total = observation.attempted_count, observation.total_count
        interval = max(1, total // 100)
        if (
            completed == total
            or completed == 1
            or completed - self._last_progress >= interval
        ):
            self._last_progress = completed
            self.progressed.emit(completed, total)

    def run(self) -> None:  # pragma: no cover - Qt signal behavior is tested
        """Execute the optimizer and publish only a complete immutable result."""
        document = self._document
        total = document.request.candidate_budget * document.request.ensemble_size
        builder = CapabilityObservationEnsembleBuilder(
            document.request.target, total, document.profile.provenance
        )
        try:
            evaluator = make_capability_flight_evaluator(
                document.profile, document.request, document.evaluator_config
            )
            hooks = CapabilityOptimizationHooks(
                observation_sink=lambda item: self._accept(builder, item),
                should_cancel=self._cancel_event.is_set,
            )
            result = optimize_capability(
                document.profile, document.request, evaluator, hooks=hooks
            )
            dataset = builder.build()
        except CapabilityOptimizationCancelled as exc:
            self.cancelled.emit(exc.attempted_count, exc.total_count)
        except Exception as exc:  # noqa: BLE001 - surface worker failures in UI
            logger.warning("capability optimization failed: %s", exc)
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result, dataset)


__all__ = ["CapabilityOptimizationWorker"]
