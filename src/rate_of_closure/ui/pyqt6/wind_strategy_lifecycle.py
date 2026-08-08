"""Fail-closed result invalidation and worker teardown for wind strategy UI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtGui import QCloseEvent

from rate_of_closure.ui.pyqt6.wind_strategy_launch import WindStrategyLaunchContext

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QComboBox, QLabel, QPushButton, QTableWidget

    from rate_of_closure.ui.pyqt6.plot_canvas_pane import PlotCanvasPane
    from rate_of_closure.ui.pyqt6.wind_strategy_worker import WindStrategyWorker
    from rate_of_closure.variation.scalar_ensemble_contract import (
        ScalarEnsembleDataset,
    )
    from shared.python.swing_sim.flight import StrategyAnalysisRequest

logger = logging.getLogger(__name__)
_SHUTDOWN_WAIT_MS = 10_000


class WindStrategyLifecycleMixin:
    """Own invalidation and ensure a QThread cannot survive widget teardown."""

    _active_context: WindStrategyLaunchContext | None
    _context_provider: Callable[[], WindStrategyLaunchContext]
    _dataset: ScalarEnsembleDataset | None
    _worker: WindStrategyWorker | None
    _request: StrategyAnalysisRequest | None
    _summary: QTableWidget
    _x_axis: QComboBox
    _y_axis: QComboBox
    _availability: QLabel
    _basis: QLabel
    _export: QPushButton
    _plot: PlotCanvasPane
    _status: QLabel

    if TYPE_CHECKING:

        def _set_running(self, running: bool) -> None: ...

    def _invalidate_settings(self, *_args: object) -> None:
        self.invalidate_result("Wind inputs changed")

    def _check_context(self) -> None:
        if self._active_context is None:
            return
        try:
            current = self._context_provider()
        except (TypeError, ValueError):
            self.invalidate_result("Current target is invalid")
            return
        if current != self._active_context:
            self.invalidate_result("Launch, target, or flight model changed")

    def invalidate_result(self, reason: str) -> None:
        """Cancel work and remove results when any consumed factor changes."""
        if self._request is None and self._dataset is None:
            return
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        self._active_context = None
        self._clear_result()
        self._status.setText(f"{reason} — rerun wind strategy analysis.")

    def _clear_result(self) -> None:
        self._request = None
        self._dataset = None
        self._summary.setRowCount(0)
        self._x_axis.clear()
        self._y_axis.clear()
        self._availability.setText("No current ensemble result.")
        self._basis.setText("Calculation basis: no current result.")
        self._export.setEnabled(False)
        self._plot.clear()

    def stop(self) -> None:
        """Cancel and join any running worker during shutdown or tests."""
        worker = self._worker
        if worker is None:
            return
        self._request = None
        self._active_context = None
        worker.cancel()
        if not worker.wait(_SHUTDOWN_WAIT_MS):
            logger.warning("wind worker exceeded shutdown grace period; joining")
            worker.wait()
        self._worker = None
        self._set_running(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Stop the worker before Qt tears down owned widgets."""
        self.stop()
        super().closeEvent(event)  # type: ignore[misc]


__all__ = ["WindStrategyLifecycleMixin"]
