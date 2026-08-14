"""Imported-job regional-ground execution workspace for PyQt6."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.regional_ground_execution_files import (
    read_regional_ground_execution_job,
    write_regional_ground_execution_job_atomic,
    write_regional_ground_execution_result_atomic,
    write_regional_ground_execution_rows_csv_atomic,
)
from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationProgress,
)
from rate_of_closure.web_authority.capability import (
    DEFAULT_UNAVAILABLE_CAPABILITY,
    AuthorityCapability,
)

from .regional_ground_execution_controller import (
    RegionalGroundExecutionController,
    RegionalGroundExecutionSubmitter,
)

Confirmation = Callable[[RegionalGroundExecutionJob], bool]
Preparation = Callable[[], RegionalGroundExecutionJob]


class RegionalGroundExecutionWorkspace(QWidget):
    """Strict import, explicit execution, and canonical result export surface."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        capability: AuthorityCapability = DEFAULT_UNAVAILABLE_CAPABILITY,
        submitter: RegionalGroundExecutionSubmitter | None = None,
        confirmation: Confirmation | None = None,
        preparation: Preparation | None = None,
    ) -> None:
        super().__init__(parent)
        if type(capability) is not AuthorityCapability:
            raise TypeError("capability must be exact")
        if submitter is not None and not callable(submitter):
            raise TypeError("submitter must be callable or None")
        if confirmation is not None and not callable(confirmation):
            raise TypeError("confirmation must be callable or None")
        if preparation is not None and not callable(preparation):
            raise TypeError("preparation must be callable or None")
        if capability.regional_ground_execution != (submitter is not None):
            raise ValueError("capability and submitter availability must agree")
        self._capability = capability
        self._confirmation = confirmation or self._confirm_imported_job
        self._preparation = preparation
        self._controller = (
            None
            if submitter is None
            else RegionalGroundExecutionController(submitter, self)
        )
        self._job: RegionalGroundExecutionJob | None = None
        self._prepared_from_editors = False
        self._prepared_stale = False
        self._result: RegionalGroundExecutionResult | None = None
        self._recent_path: Path | None = None
        self._build_ui()
        if self._controller is not None:
            self._controller.progressed.connect(self._on_progress)
            self._controller.succeeded.connect(self._on_succeeded)
            self._controller.cancelled.connect(self._on_cancelled)
            self._controller.failed.connect(self._on_failed)
            self._controller.finished.connect(self._render_actions)
        self._render_actions()

    @property
    def current_job(self) -> RegionalGroundExecutionJob | None:
        return self._job

    @property
    def current_result(self) -> RegionalGroundExecutionResult | None:
        return self._result

    @property
    def is_running(self) -> bool:
        return self._controller is not None and self._controller.is_running

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        disclosure = QLabel(
            "Prepare a strict execution-job/v1 from current validated editors, or "
            "import one. Review its exact identity, then explicitly confirm "
            "Python-authoritative execution. Preparation never starts a study. "
            "The same immutable job may be rerun; each run is a new execution "
            "attempt, while the last complete result remains available until a "
            "new job is accepted or a later run succeeds."
        )
        disclosure.setWordWrap(True)
        disclosure.setAccessibleName("Ground study execution disclosure")
        layout.addWidget(disclosure)

        actions = QHBoxLayout()
        self.open_button = self._button(
            "Open Job…", "regionalGroundOpenJobButton", "Open execution job JSON"
        )
        self.prepare_button = self._button(
            "Prepare Current Job",
            "regionalGroundPrepareJobButton",
            "Prepare an execution job from current validated editors",
        )
        self.save_job_button = self._button(
            "Save Job As…",
            "regionalGroundSaveJobButton",
            "Save canonical execution job JSON",
        )
        self.run_button = self._button(
            "Run Accepted Study",
            "regionalGroundRunButton",
            "Run the confirmed accepted study",
        )
        self.cancel_button = self._button(
            "Cancel", "regionalGroundCancelButton", "Cancel running ground study"
        )
        for button in (
            self.prepare_button,
            self.open_button,
            self.save_job_button,
            self.run_button,
            self.cancel_button,
        ):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.prepare_button.clicked.connect(self.prepare_current_job)
        self.open_button.clicked.connect(self.open_job)
        self.save_job_button.clicked.connect(self.save_job_as)
        self.run_button.clicked.connect(self.run_imported_job)
        self.cancel_button.clicked.connect(self.cancel)

        evidence = QGroupBox("Accepted authority")
        form = QFormLayout(evidence)
        self.job_label = QLabel("No execution job loaded.")
        self.job_label.setWordWrap(True)
        self.job_label.setAccessibleName("Accepted execution job evidence")
        form.addRow("Job", self.job_label)
        self.capability_label = QLabel(self._capability.detail)
        self.capability_label.setWordWrap(True)
        self.capability_label.setAccessibleName("Ground execution capability")
        form.addRow("Authority", self.capability_label)
        layout.addWidget(evidence)

        self.progress = QProgressBar()
        self.progress.setObjectName("regionalGroundExecutionProgress")
        self.progress.setAccessibleName("Ground study accepted trial progress")
        self.progress.setRange(0, 1)
        layout.addWidget(self.progress)
        self.status_label = QLabel("Open a strict execution job to begin.")
        self.status_label.setObjectName("regionalGroundExecutionStatus")
        self.status_label.setAccessibleName("Ground study execution status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        exports = QHBoxLayout()
        self.save_result_button = self._button(
            "Save Result As…",
            "regionalGroundSaveResultButton",
            "Save canonical job-bound result JSON",
        )
        self.export_csv_button = self._button(
            "Export Rows CSV…",
            "regionalGroundExportCsvButton",
            "Export every retained scalar row as CSV",
        )
        self.save_result_button.clicked.connect(self.save_result_as)
        self.export_csv_button.clicked.connect(self.export_rows_csv)
        exports.addWidget(self.save_result_button)
        exports.addWidget(self.export_csv_button)
        exports.addStretch(1)
        layout.addLayout(exports)
        layout.addStretch(1)

    @staticmethod
    def _button(text: str, object_name: str, accessible_name: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setAccessibleName(accessible_name)
        button.setToolTip(accessible_name)
        return button

    def accept_job(
        self,
        job: RegionalGroundExecutionJob,
        *,
        source_name: str,
        prepared_from_editors: bool = False,
    ) -> None:
        """Atomically replace the accepted job after complete validation."""
        if type(job) is not RegionalGroundExecutionJob:
            raise TypeError("job must be an exact RegionalGroundExecutionJob")
        job.__post_init__()
        if self.is_running:
            raise RuntimeError("cannot replace the execution job while running")
        if type(prepared_from_editors) is not bool:
            raise TypeError("prepared_from_editors must be an exact bool")
        self._job = job
        self._prepared_from_editors = prepared_from_editors
        self._prepared_stale = False
        self._result = None
        self.progress.setRange(0, job.execution_options.max_trials)
        self.progress.setValue(0)
        flight_settings = ", ".join(
            f"{name}={value:g}" for name, value in sorted(job.flight.settings.items())
        )
        self.job_label.setText(
            f"{job.job_id} · plan {job.qualified_regional_plan.request_id} · "
            f"surface {job.qualified_regional_plan.base_surface.surface_id} · "
            f"{job.flight.model_id} {job.flight.model_version} · "
            f"{job.provenance.producer} {job.provenance.producer_version} · "
            f"source {job.provenance.source_revision} · input {job.input_sha256} · "
            f"qualified plan {job.qualified_plan_sha256} · job {job.job_sha256} · "
            f"{job.execution_options.max_trials} trials · capture "
            f"{job.capture_speed_m_s:g} m/s · flight settings [{flight_settings}] · "
            f"transfer max {job.transfer.max_time_s:g} s at "
            f"{job.transfer.output_interval_s:g} s · calibration "
            f"{job.transfer.calibration.kind.value} "
            f"({job.transfer.calibration.confidence:g} confidence) · regional "
            f"step {job.regional_execution_options.settings.integration_step_s:g} s, "
            f"{job.regional_execution_options.settings.max_steps} steps, "
            f"{job.regional_execution_options.settings.max_surface_transitions} "
            "surface transitions"
        )
        self._set_status(f"Loaded {source_name}. No physics executed.", "ready")
        self._render_actions()

    def open_job(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Regional-Ground Execution Job",
            self._initial_location(),
            "JSON files (*.json)",
        )
        if not selected:
            return
        path = Path(selected)
        try:
            job = read_regional_ground_execution_job(path)
            self.accept_job(job, source_name=path.name)
        except (OSError, TypeError, ValueError, RuntimeError):
            self._set_status(
                "Open failed: the selected file is not a valid bounded "
                "execution-job/v1 document. Prior accepted evidence was preserved.",
                "error",
            )
            return
        self._recent_path = path

    def prepare_current_job(self) -> None:
        """Prepare and accept one exact editor snapshot without starting physics."""
        preparation = self._preparation
        if preparation is None:
            self._set_status(
                "Preparation unavailable: no qualified current-editor authority is "
                "injected. Prior accepted evidence was preserved.",
                "error",
            )
            self._render_actions()
            return
        if self.is_running:
            self._set_status(
                "Preparation unavailable while a study is running.", "error"
            )
            return
        try:
            candidate = preparation()
            self.accept_job(
                candidate,
                source_name="current validated editors",
                prepared_from_editors=True,
            )
        except Exception:
            self._set_status(
                "Preparation failed: current simulation, variation, and surface "
                "editors must provide one compatible validated snapshot. Prior "
                "accepted evidence was preserved.",
                "error",
            )
            self._render_actions()
            return
        self._set_status(
            "Prepared current validated editors. No physics executed; review the "
            "identity and use Run Accepted Study separately.",
            "ready",
        )

    def invalidate_prepared_job(self) -> None:
        """Make a prepared snapshot unrunnable after any owning editor changes."""
        if not self._prepared_from_editors or self._job is None:
            return
        self._prepared_stale = True
        self._set_status(
            "Stale prepared job: an owning editor changed. The frozen preview was "
            "preserved; prepare again before running.",
            "error",
        )
        self._render_actions()

    def run_imported_job(self) -> None:
        try:
            job, controller = self._require_runnable()
            confirmed = self._confirmation(job)
            if type(confirmed) is not bool:
                raise TypeError("confirmation must return an exact bool")
        except Exception:
            self._set_status(
                "Run unavailable: executable authority or confirmation failed.",
                "error",
            )
            self._render_actions()
            return
        if not confirmed:
            self._set_status(
                "Run cancelled before submission; no physics executed.", "ready"
            )
            return
        self.progress.setValue(0)
        self._set_status(
            f"Running {job.execution_options.max_trials} accepted trials…", "running"
        )
        self._render_actions(force_running=True)
        try:
            controller.submit(job)
        except Exception:
            self._set_status("Submission failed before execution started.", "error")
            self._render_actions()
            return

    def _confirm_imported_job(self, job: RegionalGroundExecutionJob) -> bool:
        answer = QMessageBox.question(
            self,
            "Run Accepted Ground Study",
            f"Run {job.execution_options.max_trials} trials from accepted job "
            f"{job.job_id} ({job.job_sha256[:12]}…)?\n\n"
            "Rerunning this immutable job starts a new attempt against the same "
            "authority. The last complete result is retained unless this run succeeds.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _require_runnable(
        self,
    ) -> tuple[RegionalGroundExecutionJob, RegionalGroundExecutionController]:
        if self._job is None:
            raise RuntimeError("no execution job is loaded")
        if self._prepared_stale:
            raise RuntimeError("prepared editor snapshot is stale")
        if self._controller is None or not self._capability.regional_ground_execution:
            raise RuntimeError(self._capability.detail)
        return self._job, self._controller

    def cancel(self) -> None:
        if self._controller is not None and self._controller.cancel():
            self._set_status(
                "Cancellation requested; waiting for the worker.", "running"
            )
            self._render_actions(force_running=True)

    def _on_progress(self, value: object) -> None:
        if type(value) is not GroundRegionalVariationProgress:
            return
        progress = cast(GroundRegionalVariationProgress, value)
        self.progress.setValue(progress.completed)
        self._set_status(
            f"Running — {progress.completed} / {progress.total} accepted trials.",
            "running",
        )

    def _on_succeeded(self, value: object) -> None:
        if type(value) is not RegionalGroundExecutionResult or self._job is None:
            self._set_status("Execution returned invalid result evidence.", "error")
            return
        result = cast(RegionalGroundExecutionResult, value)
        try:
            result.assert_matches_job(self._job)
        except (TypeError, ValueError):
            self._set_status(
                "Execution returned result evidence for a different job. "
                "The prior complete result was preserved.",
                "error",
            )
            self._render_actions()
            return
        self._result = result
        self.progress.setValue(self.progress.maximum())
        self._set_status(
            f"Succeeded — {len(result.dataset.rows)} retained rows; dataset "
            f"{result.dataset_sha256}.",
            "success",
        )
        self._render_actions()

    def _on_cancelled(self, value: object) -> None:
        if type(value) is GroundRegionalVariationCancelled:
            cancelled = cast(GroundRegionalVariationCancelled, value)
            self.progress.setValue(cancelled.completed)
            self._set_status(
                f"Cancelled — {cancelled.completed} / {cancelled.total} "
                "accepted trials.",
                "ready",
            )
        self._render_actions()

    def _on_failed(self, value: object) -> None:
        if type(value) is GroundRegionalVariationFailed:
            failure = cast(GroundRegionalVariationFailed, value)
            self.progress.setValue(failure.completed)
            self._set_status(
                f"Failed ({failure.stage.value}) — {failure.completed} / "
                f"{failure.total} accepted trials. No partial result retained; "
                "any prior complete result was preserved.",
                "error",
            )
        else:
            self._set_status(
                "Execution failed with invalid terminal evidence.", "error"
            )
        self._render_actions()

    def save_job_as(self) -> None:
        if self._job is None:
            self._set_status("Save unavailable: no execution job loaded.", "error")
            return
        self._save(
            "Save Regional-Ground Execution Job As",
            "regional-ground-execution-job.json",
            lambda path: write_regional_ground_execution_job_atomic(self._job, path),
        )

    def save_result_as(self) -> None:
        if self._result is None:
            self._set_status("Save unavailable: no complete result retained.", "error")
            return
        self._save(
            "Save Regional-Ground Execution Result As",
            "regional-ground-execution-result.json",
            lambda path: write_regional_ground_execution_result_atomic(
                self._result, path
            ),
        )

    def export_rows_csv(self) -> None:
        if self._result is None:
            self._set_status(
                "Export unavailable: no complete result retained.", "error"
            )
            return
        self._save(
            "Export Regional-Ground Execution Rows",
            "regional-ground-execution-rows.csv",
            lambda path: write_regional_ground_execution_rows_csv_atomic(
                self._result, path
            ),
            file_filter="CSV files (*.csv)",
        )

    def _save(
        self,
        title: str,
        filename: str,
        writer: Callable[[Path], bool],
        *,
        file_filter: str = "JSON files (*.json)",
    ) -> None:
        selected, _filter = QFileDialog.getSaveFileName(
            self, title, self._initial_location(filename), file_filter
        )
        if not selected:
            return
        path = Path(selected)
        try:
            writer(path)
        except (OSError, TypeError, ValueError):
            self._set_status(
                "Save failed: the destination could not be written atomically. "
                "Retained evidence was preserved.",
                "error",
            )
            return
        self._recent_path = path
        self._set_status(f"Saved {path.name} atomically.", "success")

    def _initial_location(self, filename: str = "") -> str:
        if self._recent_path is None:
            return filename
        return (
            str(self._recent_path.parent / filename)
            if filename
            else str(self._recent_path.parent)
        )

    def _set_status(self, text: str, state: str) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("state", state)
        self.status_label.setAccessibleName(
            "Ground study execution error"
            if state == "error"
            else "Ground study execution status"
        )

    def _render_actions(self, *, force_running: bool = False) -> None:
        running = force_running or self.is_running
        has_job = self._job is not None
        has_result = self._result is not None
        executable = self._capability.regional_ground_execution
        self.open_button.setEnabled(not running)
        self.save_job_button.setEnabled(has_job and not running)
        self.run_button.setEnabled(
            has_job and executable and not self._prepared_stale and not running
        )
        self.prepare_button.setEnabled(self._preparation is not None and not running)
        self.cancel_button.setEnabled(running)
        self.save_result_button.setEnabled(has_result and not running)
        self.export_csv_button.setEnabled(has_result and not running)
        reason = "" if executable else self._capability.detail
        if self._prepared_stale:
            self.run_button.setToolTip(
                "Prepared editor snapshot is stale; prepare it again before running"
            )
        else:
            self.run_button.setToolTip(reason or "Confirm and run the accepted job")
        self.prepare_button.setToolTip(
            "Prepare a job without running it"
            if self._preparation is not None
            else "No qualified current-editor preparation authority is injected"
        )
        self.cancel_button.setToolTip(
            "Request cooperative cancellation" if running else "No study is running"
        )

    def shutdown(self, timeout_ms: int = 10_000) -> None:
        """Cancel and join the owned controller before QWidget destruction."""
        if self._controller is not None:
            self._controller.shutdown(timeout_ms)


__all__ = [
    "Confirmation",
    "Preparation",
    "QFileDialog",
    "RegionalGroundExecutionWorkspace",
]
