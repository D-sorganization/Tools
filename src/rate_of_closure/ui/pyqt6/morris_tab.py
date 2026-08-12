"""Authority-backed Morris Screening workflow for the PyQt Variation module."""

from __future__ import annotations

import secrets

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.morris._response_types import MorrisResponseJob
from rate_of_closure.application.morris.presentation import (
    present_morris_factor_rows,
    present_morris_job,
    present_morris_report,
)
from rate_of_closure.application.morris.request_document import (
    build_morris_request,
    suggested_factor_drafts,
)
from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.simulation.contact import ContactMode
from rate_of_closure.ui.pyqt6.morris_controls import MorrisDesignControls
from rate_of_closure.ui.pyqt6.morris_factor_row import MorrisFactorEditor
from rate_of_closure.ui.pyqt6.morris_results import (
    RESULT_HEADERS,
    render_morris_report,
)
from rate_of_closure.ui.pyqt6.morris_worker import (
    MorrisAuthorityPort,
    MorrisCapabilityWorker,
    MorrisRunWorker,
)


class MorrisScreeningTab(QWidget):
    """Configure and present authority-owned Morris elementary effects."""

    shutdownReady = pyqtSignal()  # noqa: N815 - Qt convention

    def __init__(
        self,
        client: MorrisAuthorityPort | None,
        parent: QWidget | None = None,
        *,
        poll_interval_ms: int = 200,
    ) -> None:
        super().__init__(parent)
        if poll_interval_ms < 1:
            raise ValueError("poll_interval_ms must be positive")
        self._client = client
        self._poll_interval_ms = poll_interval_ms
        self._generation = 0
        self._authority_available = False
        self._configuration_supported = True
        self._capability_worker: MorrisCapabilityWorker | None = None
        self._worker: MorrisRunWorker | None = None
        self._retired_workers: set[MorrisRunWorker] = set()
        self._shutdown_requested = False
        self._last_job: MorrisResponseJob | None = None
        self._config = SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=113.0),
            club=get_club("Driver 10.5°"),
            source_kind="double_pendulum",
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
        )
        self._factor_rows: list[MorrisFactorEditor] = []
        self._build_ui()
        self._rebuild_factors()
        self._check_capability()

    def _build_ui(self) -> None:
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        self._design = MorrisDesignControls()
        self._trajectories = self._design.trajectories
        self._levels = self._design.levels
        self._seed = self._design.seed
        self._minimum_effects = self._design.minimum_effects
        self._workers = self._design.workers
        self._design.changed.connect(self._mark_completed_result_stale)
        controls_layout.addWidget(self._design)
        controls_layout.addWidget(self._build_factor_box(), stretch=1)
        controls_layout.addWidget(self._build_run_box())
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        controls_scroll.setMinimumWidth(535)

        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.addWidget(self._build_target_box())
        self._results = QTableWidget(0, len(RESULT_HEADERS))
        self._results.setHorizontalHeaderLabels(RESULT_HEADERS)
        self._results.setAccessibleName("Ranked Morris elementary effects")
        self._results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results.setSortingEnabled(False)
        self._results.setAlternatingRowColors(True)
        self._results.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        results_layout.addWidget(self._results, stretch=1)
        self._caveat = QLabel(
            "Results are target-specific screening effects; they do not decompose "
            "factor interactions."
        )
        self._caveat.setWordWrap(True)
        results_layout.addWidget(self._caveat)

        splitter = QSplitter()
        splitter.addWidget(controls_scroll)
        splitter.addWidget(results)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _build_factor_box(self) -> QGroupBox:
        box = QGroupBox("Ordered Factors and Bounds")
        self._factor_layout = QVBoxLayout(box)
        headings = QLabel("Use · Factor · Lower · Upper · Unit")
        headings.setToolTip("Factors retain the shared canonical registry order.")
        self._factor_layout.addWidget(headings)
        self._factor_layout.addStretch(1)
        return box

    def _build_run_box(self) -> QGroupBox:
        box = QGroupBox("Authority Run")
        layout = QVBoxLayout(box)
        buttons = QHBoxLayout()
        self._run_button = QPushButton("Run Morris Screening")
        self._run_button.setAccessibleName("Run Morris Screening")
        self._run_button.setToolTip(
            "Submit this design to the private numeric authority."
        )
        self._run_button.setEnabled(False)
        self._run_button.clicked.connect(self._start_run)
        buttons.addWidget(self._run_button, stretch=1)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setAccessibleName("Cancel Morris Screening")
        self._cancel_button.setToolTip("Request cooperative authority cancellation.")
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_run)
        buttons.addWidget(self._cancel_button)
        layout.addLayout(buttons)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setAccessibleName("Morris job progress")
        layout.addWidget(self._progress)
        self._status = QLabel("Checking local Morris authority…")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._status.setAccessibleName("Morris workflow status")
        layout.addWidget(self._status)
        return box

    def _build_target_box(self) -> QGroupBox:
        box = QGroupBox("Result Target")
        form = QFormLayout(box)
        self._target_combo = QComboBox()
        self._target_combo.setAccessibleName("Morris result target")
        self._target_combo.setToolTip(
            "Rank effects only within one output target; rankings are not mixed "
            "across units."
        )
        self._target_combo.currentIndexChanged.connect(self._render_selected_target)
        form.addRow("Target", self._target_combo)
        self._target_detail = QLabel("Run a study to choose a target.")
        self._target_detail.setWordWrap(True)
        form.addRow("Convention", self._target_detail)
        return box

    def _check_capability(self) -> None:
        if self._client is None:
            self._status.setText(
                "Morris Screening unavailable: launch the standalone app with the "
                "optional rate-morris-authority dependencies installed."
            )
            return
        worker = MorrisCapabilityWorker(self._client)
        worker.available.connect(self._capability_available)
        worker.failed.connect(self._capability_failed)
        worker.finished.connect(self._capability_finished)
        self._capability_worker = worker
        worker.start()

    def _capability_available(self, capability: object) -> None:
        available = bool(getattr(capability, "available", False))
        self._authority_available = available
        self._refresh_run_enabled()
        if not available:
            self._status.setText("Morris authority reported this workflow unavailable.")
        elif not self._configuration_supported:
            self._set_incompatible_status()
        else:
            self._set_ready_status()

    def _capability_failed(self, message: str) -> None:
        self._authority_available = False
        self._refresh_run_enabled()
        self._status.setText(message)

    def _capability_finished(self) -> None:
        worker = self.sender()
        if worker is self._capability_worker:
            self._capability_worker = None
        self._notify_shutdown_if_ready()

    def set_simulation_config(self, config: SimulationConfig) -> None:
        """Adopt the exact current simulation and rebuild its applicable factors."""
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")
        self._invalidate_active_run()
        self._config = config
        self._rebuild_factors()
        self._clear_results()
        self._configuration_supported = self._validate_current_config()
        busy = self._run_workers_running()
        self._set_running(busy)
        self._refresh_run_enabled()
        if self._configuration_supported and busy:
            self._status.setText(
                "Simulation changed; cancelling the prior Morris study before "
                "another run can start."
            )
        elif self._configuration_supported and self._authority_available:
            self._set_ready_status()

    def _validate_current_config(self) -> bool:
        try:
            build_morris_request(
                self._config,
                tuple(row.draft() for row in self._factor_rows),
                request_id="pyqt-configuration-probe",
                trajectories=self._trajectories.value(),
                levels=self._levels.value(),
                seed=self._seed.value(),
                minimum_effects=self._minimum_effects.value(),
                worker_count=self._workers.value(),
            )
        except (TypeError, ValueError):
            self._set_incompatible_status()
            return False
        return True

    def _set_incompatible_status(self) -> None:
        self._status.setText(
            "Current simulation is not authority-compatible. Select a "
            "double-pendulum, fixed-ball-contact setup in Simulation; custom "
            "torque/run semantics are intentionally never discarded."
        )

    def _set_ready_status(self) -> None:
        self._status.setText(
            "Authority ready. Review factor bounds, then run Morris Screening."
        )

    def simulation_config(self) -> SimulationConfig:
        """Return the exact currently displayed base simulation."""
        return self._config

    def set_simulation_unavailable(self, message: str) -> None:
        """Fail closed while the Simulation editor cannot build an exact base."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be nonempty")
        self._invalidate_active_run()
        self._clear_results()
        self._configuration_supported = False
        self._set_running(self._run_workers_running())
        self._status.setText(message)

    def _rebuild_factors(self) -> None:
        while self._factor_rows:
            row = self._factor_rows.pop()
            self._factor_layout.removeWidget(row)
            row.deleteLater()
        drafts = suggested_factor_drafts(self._config)
        for factor in present_morris_factor_rows(self._config, drafts):
            row = MorrisFactorEditor(factor)
            self._factor_rows.append(row)
            self._factor_layout.insertWidget(self._factor_layout.count() - 1, row)
            row.changed.connect(self._mark_completed_result_stale)

    def _mark_completed_result_stale(self) -> None:
        if self._last_job is None or self._last_job.report is None:
            return
        self._clear_results()
        self._status.setText(
            "Inputs changed after the completed study; run again for current results."
        )

    def _start_run(self) -> None:
        if self._client is None or self._worker is not None:
            return
        try:
            request = build_morris_request(
                self._config,
                tuple(row.draft() for row in self._factor_rows),
                request_id=f"pyqt-{secrets.token_hex(12)}",
                trajectories=self._trajectories.value(),
                levels=self._levels.value(),
                seed=self._seed.value(),
                minimum_effects=self._minimum_effects.value(),
                worker_count=self._workers.value(),
            )
        except (TypeError, ValueError) as exc:
            self._status.setText(f"Cannot run Morris Screening: {exc}")
            return
        self._generation += 1
        self._shutdown_requested = False
        self._clear_results()
        worker = MorrisRunWorker(
            self._client,
            request,
            self._generation,
            self._poll_interval_ms,
        )
        worker.jobChanged.connect(self._accept_job)
        worker.failed.connect(self._accept_failure)
        worker.finished.connect(self._worker_finished)
        self._worker = worker
        self._set_running(True)
        self._status.setText("Submitting Morris study…")
        worker.start()

    def _cancel_run(self) -> None:
        if self._worker is None:
            return
        self._worker.request_cancel()
        self._cancel_button.setEnabled(False)
        self._status.setText("Cancellation requested…")

    def _accept_job(self, generation: int, job: object) -> None:
        if generation != self._generation or not isinstance(job, MorrisResponseJob):
            return
        self._last_job = job
        state = present_morris_job(job)
        self._progress.setRange(0, max(state.total_samples, 1))
        self._progress.setValue(state.completed_samples)
        self._status.setText(state.message)
        self._cancel_button.setEnabled(state.can_cancel)
        if state.can_present_results and job.report is not None:
            self._populate_targets(job)

    def _accept_failure(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._status.setText(message)
        self._set_running(False)

    def _worker_finished(self) -> None:
        worker = self.sender()
        was_retired = worker in self._retired_workers
        if worker is self._worker:
            self._worker = None
        if isinstance(worker, MorrisRunWorker):
            self._retired_workers.discard(worker)
        busy = self._run_workers_running()
        self._set_running(busy)
        if (
            not busy
            and was_retired
            and not self._shutdown_requested
            and self._last_job is None
            and self._authority_available
            and self._configuration_supported
        ):
            self._set_ready_status()
        self._notify_shutdown_if_ready()

    def _set_running(self, running: bool) -> None:
        busy = running or self._run_workers_running()
        self._run_button.setEnabled(
            not busy
            and not self._shutdown_requested
            and self._authority_available
            and self._configuration_supported
        )
        self._cancel_button.setEnabled(
            self._worker is not None and not self._shutdown_requested
        )
        editable = not busy and not self._shutdown_requested
        self._design.set_editable(editable)
        for row in self._factor_rows:
            row.setEnabled(editable)

    def _refresh_run_enabled(self) -> None:
        running = self._run_workers_running()
        self._run_button.setEnabled(
            not running
            and not self._shutdown_requested
            and self._authority_available
            and self._configuration_supported
        )

    def _run_workers_running(self) -> bool:
        return bool(
            (self._worker and self._worker.isRunning())
            or any(worker.isRunning() for worker in tuple(self._retired_workers))
        )

    def _invalidate_active_run(self) -> None:
        self._generation += 1
        worker = self._worker
        if worker is not None:
            worker.request_cancel()
            self._retired_workers.add(worker)
            self._worker = None

    def _clear_results(self) -> None:
        self._last_job = None
        self._target_combo.clear()
        self._results.setRowCount(0)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._target_detail.setText("Run a study to choose a target.")
        self._caveat.setText(
            "Results are target-specific screening effects; they do not decompose "
            "factor interactions."
        )

    def _populate_targets(self, job: MorrisResponseJob) -> None:
        assert job.report is not None
        names = tuple(dict.fromkeys(item.target.name for item in job.report.estimates))
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        for name in names:
            presentation = present_morris_report(job.report, name)
            self._target_combo.addItem(presentation.target.label, name)
        self._target_combo.blockSignals(False)
        if names:
            self._target_combo.setCurrentIndex(0)
            self._render_selected_target()
        self._caveat.setText(job.report.interaction_caveat)

    def _render_selected_target(self, *_args: object) -> None:
        job = self._last_job
        target = self._target_combo.currentData()
        if job is None or job.report is None or not isinstance(target, str):
            return
        presentation = present_morris_report(job.report, target)
        render_morris_report(presentation, self._results, self._target_detail)

    def has_running_workers(self) -> bool:
        """Return whether any retained transport thread is still executing."""
        workers = tuple(self._retired_workers)
        return bool(
            (self._capability_worker and self._capability_worker.isRunning())
            or (self._worker and self._worker.isRunning())
            or any(worker.isRunning() for worker in workers)
        )

    def stop(self) -> bool:
        """Request nonblocking shutdown and retain every running QThread."""
        self._shutdown_requested = True
        self._invalidate_active_run()
        self._set_running(False)
        complete = not self.has_running_workers()
        if complete:
            self._shutdown_requested = False
        return complete

    def _notify_shutdown_if_ready(self) -> None:
        if self._shutdown_requested and not self.has_running_workers():
            self._shutdown_requested = False
            self.shutdownReady.emit()


__all__ = ["MorrisScreeningTab"]
