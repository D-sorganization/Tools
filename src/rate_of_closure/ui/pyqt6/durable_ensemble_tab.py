"""Authority-backed non-materializing ensemble workflow for PyQt6."""

from __future__ import annotations

import secrets
from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.durable_ensemble.contracts import (
    DurableEnsembleJobEnvelope,
    durable_ensemble_request_document,
    parse_durable_ensemble_request,
)
from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.ui.pyqt6.durable_ensemble_worker import (
    DurableEnsembleAuthorityPort,
    DurableEnsembleCapabilityWorker,
    DurableEnsembleRunWorker,
)
from rate_of_closure.variation.plot_labels import OUTPUT_LABELS
from shared.python.swing_sim.variation import VariationPlan


class DurableEnsembleTab(QWidget):
    """Run bounded ensembles while retaining only durable chunks and moments."""

    shutdownReady = pyqtSignal()  # noqa: N815 - Qt convention

    def __init__(
        self,
        client: DurableEnsembleAuthorityPort | None,
        plan_provider: Callable[[], VariationPlan],
        parent: QWidget | None = None,
        *,
        poll_interval_ms: int = 200,
    ) -> None:
        super().__init__(parent)
        if not callable(plan_provider) or poll_interval_ms < 1:
            raise ValueError("plan provider and positive polling interval are required")
        self._client = client
        self._plan_provider = plan_provider
        self._poll_interval_ms = poll_interval_ms
        self._config = SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=113.0),
            club=get_club("Driver 10.5°"),
            source_kind="double_pendulum",
        )
        self._authority_available = False
        self._configuration_available = True
        self._generation = 0
        self._worker: DurableEnsembleRunWorker | None = None
        self._capability_worker: DurableEnsembleCapabilityWorker | None = None
        self._shutdown_requested = False
        self._build_ui()
        self._check_capability()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        overview = QLabel(
            "Execute the plan configured under Monte Carlo & Dispersion through "
            "the local Python authority. Results are verified model-scenario "
            "moments, not human evidence or coaching recommendations."
        )
        overview.setWordWrap(True)
        layout.addWidget(overview)
        layout.addWidget(self._build_controls())
        self._results = QTableWidget(0, 5)
        self._results.setHorizontalHeaderLabels(
            ("Output", "Available", "Mean", "Sample SD", "Unit")
        )
        self._results.setAccessibleName("Durable ensemble incremental moments")
        self._results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._results.setAlternatingRowColors(True)
        layout.addWidget(self._results, stretch=1)

    def _build_controls(self) -> QGroupBox:
        box = QGroupBox("Authority Archive and Lifecycle")
        layout = QVBoxLayout(box)
        form = QFormLayout()
        self._archive_id = QLineEdit("pyqt-durable-ensemble")
        self._archive_id.setAccessibleName("Durable ensemble archive identifier")
        self._archive_id.setToolTip(
            "Stable identifier for the resumable authority-owned archive."
        )
        form.addRow("Archive ID", self._archive_id)
        self._chunk_size = QSpinBox()
        self._chunk_size.setRange(1, 256)
        self._chunk_size.setValue(8)
        self._chunk_size.setAccessibleName("Durable ensemble chunk size")
        self._chunk_size.setToolTip(
            "Number of trials committed atomically in each archive chunk."
        )
        form.addRow("Trials per chunk", self._chunk_size)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self._run_button = QPushButton("Run or Resume Durable Analysis")
        self._run_button.setEnabled(False)
        self._run_button.setToolTip(
            "Start a new analysis or resume the verified committed prefix."
        )
        self._run_button.clicked.connect(self._start)
        buttons.addWidget(self._run_button, stretch=1)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._cancel_button.setToolTip(
            "Request cooperative cancellation without deleting committed chunks."
        )
        self._cancel_button.clicked.connect(self._cancel)
        buttons.addWidget(self._cancel_button)
        layout.addLayout(buttons)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setAccessibleName("Durable ensemble progress")
        layout.addWidget(self._progress)
        self._status = QLabel("Checking local durable ensemble authority…")
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._status)
        return box

    def _check_capability(self) -> None:
        if self._client is None:
            self._status.setText(
                "Durable analysis unavailable outside the standalone local authority."
            )
            return
        worker = DurableEnsembleCapabilityWorker(self._client)
        worker.available.connect(self._capability_available)
        worker.failed.connect(self._capability_failed)
        worker.finished.connect(self._capability_finished)
        self._capability_worker = worker
        worker.start()

    def _capability_available(self, capability: object) -> None:
        self._authority_available = bool(getattr(capability, "available", False))
        self._refresh_enabled()
        self._status.setText(
            "Authority ready. Configure a global swing perturbation plan, then run."
            if self._authority_available
            else "Durable ensemble authority reported this workflow unavailable."
        )

    def _capability_failed(self, message: str) -> None:
        self._authority_available = False
        self._refresh_enabled()
        self._status.setText(message)

    def _capability_finished(self) -> None:
        if self.sender() is self._capability_worker:
            self._capability_worker = None
        self._notify_shutdown()

    def set_simulation_config(self, config: SimulationConfig) -> None:
        """Adopt the exact base used to author the authority request."""
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")
        self._invalidate_active()
        self._config = config
        self._configuration_available = True
        self._refresh_enabled()

    def set_simulation_unavailable(self, message: str) -> None:
        """Fail closed when the Simulation editor has no exact base."""
        self._invalidate_active()
        self._configuration_available = False
        self._refresh_enabled()
        self._status.setText(message)

    def _start(self) -> None:
        if self._client is None or self._worker is not None:
            return
        try:
            plan = self._plan_provider()
            document = durable_ensemble_request_document(
                f"pyqt-{secrets.token_hex(8)}",
                self._archive_id.text(),
                plan,
                self._config,
                chunk_size=self._chunk_size.value(),
            )
            request = parse_durable_ensemble_request(document)
        except Exception as exc:
            self._status.setText(f"Cannot start durable analysis: {exc}")
            return
        self._generation += 1
        worker = DurableEnsembleRunWorker(
            self._client, request, self._generation, self._poll_interval_ms
        )
        worker.jobChanged.connect(self._job_changed)
        worker.failed.connect(self._run_failed)
        worker.finished.connect(self._run_finished)
        self._worker = worker
        self._set_running(True)
        self._status.setText("Submitting or verifying the durable archive…")
        worker.start()

    def _job_changed(self, generation: int, value: object) -> None:
        if generation != self._generation or not isinstance(
            value, DurableEnsembleJobEnvelope
        ):
            return
        self._progress.setRange(0, value.total_trials)
        self._progress.setValue(value.completed_trials)
        if value.evidence is not None:
            self._render(value)
        if value.status in {"completed", "cancelled", "failed"}:
            suffix = f": {value.error}" if value.error else ""
            self._status.setText(
                f"Durable analysis {value.status}; retained "
                f"{value.completed_trials}/{value.total_trials} trials{suffix}."
            )

    def _render(self, job: DurableEnsembleJobEnvelope) -> None:
        assert job.evidence is not None
        rows = job.evidence.output_moments
        self._results.setRowCount(len(rows))
        for row, moment in enumerate(rows):
            values = (
                OUTPUT_LABELS[moment.name],
                str(moment.available_count),
                "—" if moment.mean is None else f"{moment.mean:.6g}",
                "—" if moment.sample_std is None else f"{moment.sample_std:.6g}",
                moment.unit,
            )
            for column, text in enumerate(values):
                self._results.setItem(row, column, QTableWidgetItem(text))

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_cancel()
            self._status.setText(
                "Cancellation requested; committed chunks are retained."
            )

    def _run_failed(self, generation: int, message: str) -> None:
        if generation == self._generation:
            self._status.setText(message)

    def _run_finished(self) -> None:
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
        if isinstance(worker, DurableEnsembleRunWorker):
            worker.deleteLater()
        self._set_running(False)
        self._notify_shutdown()

    def _invalidate_active(self) -> None:
        self._generation += 1
        if self._worker is not None:
            self._worker.request_cancel()

    def _set_running(self, running: bool) -> None:
        self._cancel_button.setEnabled(running)
        self._archive_id.setEnabled(not running)
        self._chunk_size.setEnabled(not running)
        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        self._run_button.setEnabled(
            self._worker is None
            and self._authority_available
            and self._configuration_available
        )

    def stop(self) -> bool:
        """Request cancellation and report whether all transport is already idle."""
        self._shutdown_requested = True
        if self._worker is not None:
            self._worker.request_cancel()
        ready = self._worker is None and self._capability_worker is None
        if ready:
            self.shutdownReady.emit()
        return ready

    def _notify_shutdown(self) -> None:
        if (
            self._shutdown_requested
            and self._worker is None
            and self._capability_worker is None
        ):
            self.shutdownReady.emit()


__all__ = ["DurableEnsembleTab"]
