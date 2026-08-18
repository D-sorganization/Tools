"""Status-label and action-enablement rendering for the execution workspace."""

from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QProgressBar, QPushButton

from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)
from rate_of_closure.web_authority.capability import AuthorityCapability

from .regional_ground_execution_controller import RegionalGroundExecutionController


class RegionalGroundExecutionStatusMixin:
    """Render the accessible status line and every action's enabled state.

    Split out of the workspace for the 500 LOC budget. Kept as a mixin because
    every method here reads the workspace's own widgets and retained evidence.
    """

    # Supplied by the concrete workspace; declared so this module type-checks
    # on its own under CI's ``--follow-imports=skip``.
    status_label: QLabel
    progress: QProgressBar
    open_button: QPushButton
    save_job_button: QPushButton
    run_button: QPushButton
    prepare_button: QPushButton
    cancel_button: QPushButton
    save_result_button: QPushButton
    export_csv_button: QPushButton
    is_running: bool
    _job: RegionalGroundExecutionJob | None
    _result: RegionalGroundExecutionResult | None
    _capability: AuthorityCapability
    _controller: RegionalGroundExecutionController | None
    _prepared_stale: bool
    _preparation: object | None

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


__all__ = ["RegionalGroundExecutionStatusMixin"]
