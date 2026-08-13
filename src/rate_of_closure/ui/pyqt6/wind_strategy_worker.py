"""Responsive QThread wrapper for shared wind-strategy analysis."""

from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from shared.python.swing_sim.flight import StrategyAnalysisRequest
from shared.python.swing_sim.flight.wind_strategy import (
    WindStrategyCancelledError,
    WindStrategyProgress,
    analyze_wind_strategies,
)

logger = logging.getLogger(__name__)


class WindStrategyWorker(QThread):
    """Run one immutable request off the GUI thread with exact progress."""

    progressed = pyqtSignal(int, int)  # noqa: N815 - Qt signal convention
    succeeded = pyqtSignal(object)  # noqa: N815
    cancelled = pyqtSignal()  # noqa: N815
    failed = pyqtSignal(str)  # noqa: N815

    def __init__(self, request: StrategyAnalysisRequest) -> None:
        super().__init__()
        if not isinstance(request, StrategyAnalysisRequest):
            raise TypeError("request must be a StrategyAnalysisRequest")
        self._request = request
        self._cancel_event = threading.Event()

    @property
    def cancel_event(self) -> threading.Event:
        """Return the event passed directly into the shared core."""
        return self._cancel_event

    def cancel(self) -> None:
        """Request cooperative cancellation at the next outcome boundary."""
        self._cancel_event.set()

    def _progress(self, report: WindStrategyProgress) -> None:
        self.progressed.emit(report.completed_outcomes, report.total_outcomes)

    def run(self) -> None:  # pragma: no cover - signal behavior is tested
        """Execute and translate typed completion states into Qt signals."""
        try:
            result = analyze_wind_strategies(
                self._request,
                progress_cb=self._progress,
                cancel_event=self._cancel_event,
            )
        except WindStrategyCancelledError:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - expose worker failures in UI
            logger.warning("wind strategy analysis failed: %s", exc)
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)


__all__ = ["WindStrategyWorker"]
