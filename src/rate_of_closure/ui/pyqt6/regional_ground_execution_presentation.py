"""Read-only PyQt presentation of regional-ground controller evidence."""

from __future__ import annotations

from PyQt6.QtWidgets import QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from rate_of_closure.application.regional_ground_execution_job import (
    RegionalGroundExecutionJob,
)
from rate_of_closure.application.regional_ground_execution_presentation import (
    RegionalGroundExecutionPresentation,
)
from rate_of_closure.application.regional_ground_execution_result import (
    RegionalGroundExecutionResult,
)
from rate_of_closure.variation.regional_ground_variation_control import (
    GroundRegionalVariationCancelled,
    GroundRegionalVariationFailed,
    GroundRegionalVariationProgress,
)
from rate_of_closure.web_authority.capability import AuthorityCapability

from .regional_ground_execution_controller import RegionalGroundExecutionController


class RegionalGroundExecutionPresentationPanel(QWidget):
    """Compact disabled controls and exact read-only execution evidence."""

    def __init__(
        self,
        job: RegionalGroundExecutionJob,
        capability: AuthorityCapability,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._presentation = RegionalGroundExecutionPresentation.initial(
            job, capability
        )
        self._controller: RegionalGroundExecutionController | None = None
        self.summary_label = QLabel(self)
        self.disabled_label = QLabel(self)
        self.status_label = QLabel(self)
        self.run_button = QPushButton("Run study", self)
        self.cancel_button = QPushButton("Cancel study", self)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        explanation = self._presentation.disabled_detail
        self.run_button.setToolTip(explanation)
        self.cancel_button.setToolTip(explanation)
        form = QFormLayout()
        form.addRow("Evidence", self.summary_label)
        form.addRow("Availability", self.disabled_label)
        form.addRow("Status", self.status_label)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.run_button)
        layout.addWidget(self.cancel_button)
        self._render()

    def bind_controller(self, controller: RegionalGroundExecutionController) -> None:
        """Observe one existing controller without exposing execution controls."""
        if type(controller) is not RegionalGroundExecutionController:
            raise TypeError("controller must be exact")
        if self._controller is not None:
            raise RuntimeError("presentation panel is already bound")
        controller.progressed.connect(self.present_progress)
        controller.succeeded.connect(self.present_result)
        controller.cancelled.connect(self.present_cancelled)
        controller.failed.connect(self.present_failure)
        self._controller = controller

    def present_progress(self, progress: GroundRegionalVariationProgress) -> None:
        """Present exact progress received from an existing controller."""
        self._presentation = self._presentation.with_progress(progress)
        self._render()

    def present_cancel_requested(self) -> None:
        """Present a controller cancellation-requested state."""
        self._presentation = self._presentation.with_cancel_requested()
        self._render()

    def present_cancelled(self, terminal: GroundRegionalVariationCancelled) -> None:
        """Present an exact controller cancellation terminal."""
        self._presentation = self._presentation.with_cancelled(terminal)
        self._render()

    def present_failure(self, terminal: GroundRegionalVariationFailed) -> None:
        """Present only the typed controller failure stage and counts."""
        self._presentation = self._presentation.with_failure(terminal)
        self._render()

    def present_result(self, result: RegionalGroundExecutionResult) -> None:
        """Present a complete identity-bound controller result."""
        self._presentation = self._presentation.with_result(result)
        self._render()

    def _render(self) -> None:
        """Render one immutable presentation without invoking any control."""
        item = self._presentation
        summary = item.summary
        self.summary_label.setText(
            f"{summary.schema_version} · {summary.model_id} "
            f"{summary.model_version} · {summary.producer} "
            f"{summary.producer_version} · {summary.source_revision} · "
            f"input {summary.input_sha256}"
        )
        self.disabled_label.setText(
            f"{item.disabled_reason_code}: {item.disabled_detail}"
        )
        state = item.state.value.replace("_", " ").title()
        if item.failure_stage is not None:
            state = f"Failed ({item.failure_stage})"
        self.status_label.setText(
            f"{state} — {item.completed} / {item.total} accepted trials"
        )


__all__ = ["RegionalGroundExecutionPresentationPanel"]
