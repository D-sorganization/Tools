"""Generation-bound background computation for the PyQt Plots workspace."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from PyQt6.QtCore import Qt, QThread, pyqtBoundSignal, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QHideEvent, QShowEvent

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.plotting import PlotData, PlotSpec
from rate_of_closure.simulation import SimulationConfig, SimulationRun, run_simulation

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QLabel, QListWidget

    from rate_of_closure.ui.pyqt6.plot_canvas_pane import PlotCanvasPane
logger = logging.getLogger(__name__)
_DEFAULT_CLUB = "Driver 10.5°"
PlotExecutor = Callable[[PlotSpec, SimulationRun, Callable[[], bool] | None], PlotData]


@dataclass(frozen=True)
class PlotComputationOutcome:
    """One independent pane's complete data or bounded failure."""

    row: int
    data: PlotData | None
    error: str | None


class PlotComputeWorker(QThread):
    """Compute immutable plot data without blocking Qt layout or input."""

    succeeded = pyqtSignal(int, object, object)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        generation: int,
        requests: tuple[tuple[int, PlotSpec], ...],
        reference_run: SimulationRun | None,
        scenario: ImpactScenario,
        executor: PlotExecutor,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.requests = requests
        self._requests = requests
        self._reference_run = reference_run
        self._scenario = scenario
        self._executor = executor

    def run(self) -> None:
        """Build one reference run and every requested pane outcome."""
        try:
            run = self._reference_run or run_simulation(
                SimulationConfig(
                    scenario=self._scenario,
                    club=get_club(_DEFAULT_CLUB),
                )
            )
            outcomes: list[PlotComputationOutcome] = []
            for row, spec in self._requests:
                if self.isInterruptionRequested():
                    return
                try:
                    data = self._executor(spec, run, self.isInterruptionRequested)
                    outcomes.append(PlotComputationOutcome(row, data, None))
                except InterruptedError:
                    return
                except Exception as exc:  # noqa: BLE001 - pane-scoped status
                    outcomes.append(PlotComputationOutcome(row, None, str(exc)[:512]))
            if not self.isInterruptionRequested():
                self.succeeded.emit(self.generation, run, tuple(outcomes))
        except Exception as exc:  # noqa: BLE001 - retained-state status
            self.failed.emit(self.generation, str(exc)[:512])


class PlotWorker(Protocol):
    """Structural lifecycle shared by QThread and QProcess workers."""

    succeeded: pyqtBoundSignal
    failed: pyqtBoundSignal
    finished: pyqtBoundSignal
    requests: tuple[tuple[int, PlotSpec], ...]

    def start(self) -> None: ...
    def requestInterruption(self) -> None: ...  # noqa: N802
    def wait(self, timeout_ms: int) -> bool: ...
    def isRunning(self) -> bool: ...  # noqa: N802
    def deleteLater(self) -> None: ...  # noqa: N802


class PlotsTabComputationMixin:
    """Own worker generations while the concrete tab owns presentation."""

    if TYPE_CHECKING:
        _scenario: ImpactScenario
        _run: SimulationRun | None
        _data: PlotData | None
        _plot_panes: list[PlotCanvasPane]
        _plot_data: list[PlotData | None]
        _plot_data_current: list[bool]
        _plot_list: QListWidget
        _status: QLabel

        def isVisible(self) -> bool: ...
        def _sync_selected_pane(self) -> None: ...
        def _plot_compute_executor(self) -> PlotExecutor: ...
        def _plot_process_enabled(self) -> bool: ...

    def _init_plot_computation(self) -> None:
        self._plot_generation = 0
        self._plot_worker: PlotWorker | None = None
        self._plot_refresh_pending = False

    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Invalidate exact plot evidence for a new explorer scenario."""
        self._scenario = scenario
        self._run = None
        self._mark_plot_data_stale()
        self._request_refresh_if_visible()

    def set_run(self, run: SimulationRun) -> None:
        """Invalidate exact plot evidence for a new accepted reference run."""
        self._run = run
        self._mark_plot_data_stale()
        self._request_refresh_if_visible()

    def showEvent(self, event: QShowEvent | None) -> None:  # noqa: N802
        """Show a stable frame immediately and compute stale data off-thread."""
        super().showEvent(event)  # type: ignore[misc]
        if not all(self._plot_data_current):
            self._request_plot_refresh()

    def hideEvent(self, event: QHideEvent | None) -> None:  # noqa: N802
        """Cancel hidden work; the next show restarts exact current authority."""
        self._plot_generation += 1
        self._plot_refresh_pending = False
        if self._plot_worker is not None:
            self._plot_worker.requestInterruption()
        super().hideEvent(event)  # type: ignore[misc]

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        """Cooperatively stop the worker before Qt destroys its owner."""
        worker = self._plot_worker
        if worker is not None:
            worker.requestInterruption()
            worker.wait(3_000)
        super().closeEvent(event)  # type: ignore[misc]

    def _request_refresh_if_visible(self) -> None:
        if self.isVisible():
            self._request_plot_refresh()

    def _mark_plot_data_stale(self) -> None:
        self._plot_data_current = [False] * len(self._plot_panes)

    def _request_plot_refresh(self) -> None:
        self._plot_generation += 1
        if self._plot_worker is not None:
            self._plot_refresh_pending = True
            self._plot_worker.requestInterruption()
            return
        self._start_plot_worker(self._plot_generation)

    def _start_plot_worker(self, generation: int) -> None:
        request_items: list[tuple[int, object]] = []
        for row in range(self._plot_list.count()):
            if self._plot_data_current[row]:
                continue
            item = self._plot_list.item(row)
            if item is None:
                self._status.setText(
                    "Plot definitions are malformed; prior plots retained"
                )
                return
            request_items.append((row, item.data(Qt.ItemDataRole.UserRole)))
        requests = tuple(request_items)
        if not requests:
            self._sync_selected_pane()
            self._status.setText("")
            return
        if not all(isinstance(spec, PlotSpec) for _row, spec in requests):
            self._status.setText("Plot definitions are malformed; prior plots retained")
            return
        retained = any(data is not None for data in self._plot_data)
        self._status.setText(
            "Computing plots…" + ("; prior accepted plots retained" if retained else "")
        )
        if self._plot_process_enabled():
            from rate_of_closure.ui.pyqt6.plots_process_worker import (
                PlotComputeProcess,
                PlotProcessRequest,
            )

            try:
                worker: PlotWorker = PlotComputeProcess(
                    PlotProcessRequest(generation, requests, self._run, self._scenario)
                )
            except Exception as exc:  # noqa: BLE001 - bounded IPC preflight
                retained_suffix = (
                    "; prior accepted plots retained" if any(self._plot_data) else ""
                )
                self._status.setText(
                    "Plot computation could not start: "
                    f"{str(exc)[:512]}{retained_suffix}"
                )
                return
        else:
            worker = PlotComputeWorker(
                generation,
                requests,
                self._run,
                self._scenario,
                self._plot_compute_executor(),
            )
        worker.succeeded.connect(
            lambda emitted, run, outcomes, owner=worker: self._accept_plot_success(
                owner, emitted, run, outcomes
            )
        )
        worker.failed.connect(
            lambda emitted, message, owner=worker: self._accept_plot_failure(
                owner, emitted, message
            )
        )
        worker.finished.connect(lambda owner=worker: self._accept_plot_finished(owner))
        self._plot_worker = worker
        worker.start()

    def _accept_plot_success(
        self,
        owner: PlotWorker,
        generation: int,
        run: object,
        outcomes: object,
    ) -> None:
        if owner is not self._plot_worker or generation != self._plot_generation:
            return
        if not isinstance(run, SimulationRun) or not isinstance(outcomes, tuple):
            self._accept_plot_failure(owner, generation, "Malformed plot worker result")
            return
        if not all(
            isinstance(item, PlotComputationOutcome)
            and type(item.row) is int
            and (
                (isinstance(item.data, PlotData) and item.error is None)
                or (
                    item.data is None
                    and isinstance(item.error, str)
                    and 0 < len(item.error) <= 512
                )
            )
            for item in outcomes
        ):
            self._accept_plot_failure(owner, generation, "Malformed plot pane results")
            return
        rows = tuple(item.row for item in outcomes)
        if rows != tuple(row for row, _spec in owner.requests) or any(
            row < 0 or row >= len(self._plot_panes) for row in rows
        ):
            self._accept_plot_failure(owner, generation, "Malformed plot pane identity")
            return
        expected_specs = dict(owner.requests)
        for outcome in outcomes:
            item = self._plot_list.item(outcome.row)
            current_spec = (
                item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            )
            if current_spec != expected_specs[outcome.row] or (
                outcome.data is not None and outcome.data.spec != current_spec
            ):
                self._accept_plot_failure(
                    owner, generation, "Malformed plot pane authority"
                )
                return
        self._run = run
        errors: list[str] = []
        for outcome in outcomes:
            row = outcome.row
            item = self._plot_list.item(row)
            label = item.text() if item is not None else f"Plot {row + 1}"
            if outcome.data is None:
                retained = (
                    "; prior accepted plot retained" if self._plot_data[row] else ""
                )
                errors.append(f"{label}: {outcome.error}{retained}")
                continue
            try:
                self._plot_panes[row].render_data(outcome.data)
            except Exception as exc:  # noqa: BLE001 - pane-scoped rollback
                retained = (
                    "; prior accepted plot retained" if self._plot_data[row] else ""
                )
                errors.append(f"{label}: {str(exc)[:512]}{retained}")
                continue
            self._plot_data[row] = outcome.data
            self._plot_data_current[row] = True
        self._sync_selected_pane()
        self._status.setText("; ".join(errors))

    def _accept_plot_failure(
        self, owner: PlotWorker, generation: int, message: str
    ) -> None:
        if owner is not self._plot_worker or generation != self._plot_generation:
            return
        retained = "; prior accepted plots retained" if any(self._plot_data) else ""
        self._status.setText(f"Plot computation failed: {message[:512]}{retained}")

    def _accept_plot_finished(self, owner: PlotWorker) -> None:
        if owner is not self._plot_worker:
            return
        self._plot_worker = None
        owner.deleteLater()
        if self._plot_refresh_pending and self.isVisible():
            self._plot_refresh_pending = False
            self._start_plot_worker(self._plot_generation)

    def reference_run(self) -> SimulationRun | None:
        """Return the reference run, building it synchronously on explicit demand."""
        if self._run is None:
            try:
                self._run = run_simulation(
                    SimulationConfig(
                        scenario=self._scenario,
                        club=get_club(_DEFAULT_CLUB),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - surfaced in status
                logger.warning("reference run failed: %s", exc)
                self._status.setText(f"Reference run failed: {exc}")
        return self._run

    def refresh(self) -> None:
        """Synchronously refresh on explicit API demand; product paths use workers."""
        self._plot_generation += 1
        worker = self._plot_worker
        if worker is not None:
            worker.requestInterruption()
            worker.wait(3_000)
            self._plot_worker = None
        run = self.reference_run()
        if run is None:
            return
        errors: list[str] = []
        current_row = self._plot_list.currentRow()
        for row in range(len(self._plot_panes)):
            if not self._plot_data_current[row]:
                error = self._compute_row(row, run)
                if error:
                    errors.append(error)
            if row == current_row:
                self._data = self._plot_data[row]
        self._sync_selected_pane()
        self._status.setText("; ".join(errors))

    def _compute_row(self, row: int, run: SimulationRun) -> str | None:
        item = self._plot_list.item(row)
        if item is None:
            return None
        spec = item.data(Qt.ItemDataRole.UserRole)
        try:
            data = self._plot_compute_executor()(spec, run, None)
            self._plot_panes[row].render_data(data)
            self._plot_data[row] = data
            self._plot_data_current[row] = True
        except Exception as exc:  # noqa: BLE001 - plotting must not crash
            logger.warning("plot render failed: %s", exc)
            retained = "; prior accepted plot retained" if self._plot_data[row] else ""
            return f"{item.text()}: {exc}{retained}"
        return None

    def _on_selection_changed(self, _row: int) -> None:
        self._sync_selected_pane()
        row = self._plot_list.currentRow()
        if self.isVisible() and 0 <= row < len(self._plot_data):
            if not self._plot_data_current[row]:
                self._request_plot_refresh()
            self._data = self._plot_data[row]


__all__ = ["PlotComputeWorker", "PlotComputationOutcome", "PlotsTabComputationMixin"]
