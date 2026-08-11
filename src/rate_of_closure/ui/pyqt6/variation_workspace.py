"""Native variation authoring policy and workspace-state adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from rate_of_closure.application.workspace_variation_session import (
    VariationAnalysisExecution,
    VariationWorkspaceState,
    available_output_metrics,
)

if TYPE_CHECKING:
    from shared.python.swing_sim.variation import VariationPlan


_EXECUTION_LABELS = {
    VariationAnalysisExecution.ALL_TOGETHER: "All Enabled Together",
    VariationAnalysisExecution.INDIVIDUAL: "Each Enabled Individually (OAT)",
    VariationAnalysisExecution.BOTH: "Both",
}


class VariationWorkspaceMixin:
    """Own persisted execution/output policy beside the canonical plan editors."""

    _analysis_execution_combo: QComboBox
    _metric_checks: dict[str, QCheckBox]
    _metric_layout: QGridLayout
    _status: QLabel

    if TYPE_CHECKING:

        def build_plan(self) -> VariationPlan: ...

        def load_plan(self, plan: VariationPlan) -> None: ...

        def mode(self) -> str: ...

    def _build_workspace_policy_box(self) -> QGroupBox:
        """Build discoverable execution and output-focus controls."""
        box = QGroupBox("Analysis Policy")
        layout = QVBoxLayout(box)
        self._analysis_execution_combo = QComboBox()
        for execution, label in _EXECUTION_LABELS.items():
            self._analysis_execution_combo.addItem(label, execution)
        self._analysis_execution_combo.setCurrentIndex(
            self._analysis_execution_combo.findData(VariationAnalysisExecution.BOTH)
        )
        self._analysis_execution_combo.setToolTip(
            "Run all enabled inputs simultaneously, each input individually, or both. "
            "This workspace policy does not change the physical variation plan."
        )
        layout.addWidget(QLabel("Analysis Execution"))
        layout.addWidget(self._analysis_execution_combo)
        metric_box = QGroupBox("Selected Output Metrics")
        self._metric_layout = QGridLayout(metric_box)
        self._metric_checks = {}
        self._reset_output_metric_choices()
        metric_box.setToolTip(
            "Saved output focus for this study. Runs retain their complete canonical "
            "result contract; these selections do not fabricate or discard results."
        )
        layout.addWidget(metric_box)
        return box

    def _reset_output_metric_choices(self) -> None:
        """Replace metric controls with the current plan mode's canonical outputs."""
        if not hasattr(self, "_metric_layout"):
            return
        while self._metric_layout.count():
            item = self._metric_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._metric_checks = {}
        for index, metric in enumerate(available_output_metrics(self.mode())):
            checkbox = QCheckBox(metric)
            checkbox.setChecked(True)
            checkbox.setToolTip(f"Include {metric} in the saved output focus.")
            self._metric_layout.addWidget(checkbox, index // 2, index % 2)
            self._metric_checks[metric] = checkbox

    def analysis_execution(self) -> VariationAnalysisExecution:
        """Return the selected simultaneous/individual execution policy."""
        value = self._analysis_execution_combo.currentData()
        if not isinstance(value, VariationAnalysisExecution):
            raise TypeError("analysis execution control contains invalid data")
        return value

    def variation_workspace_state(self) -> VariationWorkspaceState:
        """Capture the fully validated authored study specification."""
        selected = tuple(
            metric
            for metric, checkbox in self._metric_checks.items()
            if checkbox.isChecked()
        )
        return VariationWorkspaceState(
            plan=self.build_plan(),
            analysis_execution=self.analysis_execution(),
            selected_output_metrics=selected,
        )

    def apply_variation_workspace_state(self, state: VariationWorkspaceState) -> None:
        """Apply one validated plan and its persisted authoring policy."""
        if not isinstance(state, VariationWorkspaceState):
            raise TypeError("state must be a VariationWorkspaceState")
        self.load_plan(state.plan)
        execution_index = self._analysis_execution_combo.findData(
            state.analysis_execution
        )
        if execution_index < 0:
            raise ValueError("analysis execution is unavailable in the native UI")
        self._analysis_execution_combo.setCurrentIndex(execution_index)
        selected = set(state.selected_output_metrics)
        for metric, checkbox in self._metric_checks.items():
            checkbox.setChecked(metric in selected)
        self._status.setText(
            "Workspace restored variation plan, execution policy, and output focus."
        )


__all__ = ["VariationWorkspaceMixin"]
