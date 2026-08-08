"""User-facing PyQt capability optimization workflow."""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.capability_workflow import (
    CapabilityWorkflowDocument,
    build_capability_workflow,
    capability_workflow_from_json,
    capability_workflow_inputs,
    capability_workflow_json,
)
from rate_of_closure.ui.pyqt6.capability_controls import CapabilityControls
from rate_of_closure.ui.pyqt6.capability_results import CapabilityResults
from rate_of_closure.ui.pyqt6.capability_worker import CapabilityOptimizationWorker
from rate_of_closure.variation.capability_observation_adapter import (
    capability_observation_ensemble_json,
)
from rate_of_closure.variation.scalar_ensemble_contract import ScalarEnsembleDataset
from rate_of_closure.variation.scalar_ensemble_io import scalar_ensemble_csv
from shared.python.swing_sim.flight.capability_result import OptimizationResult

logger = logging.getLogger(__name__)
_SHUTDOWN_WAIT_MS = 10_000
_ACTION_TOOLTIPS = (
    "Validate the captured basis and run optimization in a background thread.",
    "Request cooperative cancellation; partial results are never published.",
    "Save the complete versioned profile, target, model, and search basis.",
    "Load and strictly validate a saved capability workflow.",
    "Export every retained observation as spreadsheet-safe lossless CSV.",
    "Export every retained observation using the stable versioned JSON schema.",
)


class CapabilityOptimizationTab(QWidget):
    """Author, run, inspect, persist, and export one robust shot search."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: CapabilityOptimizationWorker | None = None
        self._document: CapabilityWorkflowDocument | None = None
        self._dataset: ScalarEnsembleDataset | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        note = QLabel(
            "Still-air carry to first ground crossing. Fixed spin is explicit and "
            "sourced; wind, bounce, roll, and total distance are not included."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        splitter = QSplitter()
        self.controls = CapabilityControls()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.controls)
        scroll.setMinimumWidth(340)
        self.results = CapabilityResults()
        self.results.setMinimumWidth(520)
        self.results.setVisible(False)
        splitter.addWidget(scroll)
        splitter.addWidget(self.results)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 640])
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(self._build_actions())
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.status = QLabel("Ready")
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        self.cancel_button.setEnabled(False)
        self.csv_button.setEnabled(False)
        self.json_button.setEnabled(False)
        self._connect_signals()

    def _build_actions(self) -> QHBoxLayout:
        actions = QHBoxLayout()
        self.run_button = QPushButton("Run Optimization")
        self.cancel_button = QPushButton("Cancel")
        self.save_button = QPushButton("Save Workflow")
        self.load_button = QPushButton("Load Workflow")
        self.csv_button = QPushButton("Export Raw CSV")
        self.json_button = QPushButton("Export Stable JSON")
        buttons = (
            self.run_button,
            self.cancel_button,
            self.save_button,
            self.load_button,
            self.csv_button,
            self.json_button,
        )
        for button, tooltip in zip(buttons, _ACTION_TOOLTIPS, strict=True):
            actions.addWidget(button)
            button.setToolTip(tooltip)
        return actions

    def _connect_signals(self) -> None:
        self.controls.changed.connect(self._invalidate)
        self.run_button.clicked.connect(self.run)
        self.cancel_button.clicked.connect(self.cancel)
        self.save_button.clicked.connect(self._save_workflow)
        self.load_button.clicked.connect(self._load_workflow)
        self.csv_button.clicked.connect(self._export_csv)
        self.json_button.clicked.connect(self._export_json)

    def _invalidate(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        self._document = None
        self._dataset = None
        self.results.setVisible(False)
        self.csv_button.setEnabled(False)
        self.json_button.setEnabled(False)
        self.status.setText("Inputs changed — run again.")

    def run(self) -> None:
        """Validate the full basis before starting a background calculation."""
        self.stop()
        try:
            document = build_capability_workflow(self.controls.inputs())
        except (TypeError, ValueError) as exc:
            self.status.setText(f"Invalid inputs: {exc}")
            return
        self._document = document
        total = document.request.candidate_budget * document.request.ensemble_size
        self.progress.setRange(0, total)
        self.progress.setValue(0)
        self.status.setText(f"Running 0/{total} model evaluations…")
        self._worker = CapabilityOptimizationWorker(document)
        self._worker.progressed.connect(self._on_progress)
        self._worker.succeeded.connect(self._on_success)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.failed.connect(self._on_failed)
        self._set_running(True)
        self._worker.start()

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _on_progress(self, completed: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(completed)
        self.status.setText(f"Running {completed}/{total} model evaluations…")

    def _on_success(
        self, result: OptimizationResult, dataset: ScalarEnsembleDataset
    ) -> None:
        self._dataset = dataset
        self.results.set_output(result, dataset)
        self.results.setVisible(True)
        self.csv_button.setEnabled(True)
        self.json_button.setEnabled(True)
        self.status.setText(f"Completed {len(dataset.rows)} observations.")
        self._set_running(False)

    def _on_cancelled(self, completed: int, total: int) -> None:
        self.status.setText(f"Cancelled after {completed}/{total} evaluations.")
        self._set_running(False)

    def _on_failed(self, message: str) -> None:
        self.status.setText(f"Optimization failed: {message}")
        self._set_running(False)

    def cancel(self) -> None:
        """Request typed cancellation without publishing partial results."""
        if self._worker is not None:
            self._worker.cancel()
            self.status.setText("Cancellation requested…")

    def stop(self) -> None:
        """Cancel and join the worker so it cannot outlive the application."""
        worker = self._worker
        if worker is None:
            return
        worker.cancel()
        if worker.isRunning() and not worker.wait(_SHUTDOWN_WAIT_MS):
            logger.warning("capability worker exceeded shutdown grace period; joining")
            worker.wait()
        self._worker = None
        self._set_running(False)

    def _save_workflow(self) -> None:
        try:
            source = capability_workflow_json(
                build_capability_workflow(self.controls.inputs())
            )
        except (TypeError, ValueError) as exc:
            self.status.setText(f"Cannot save invalid workflow: {exc}")
            return
        self._save_text("Save Capability Workflow", "capability-workflow.json", source)

    def _load_workflow(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self, "Load Capability Workflow", "", "JSON files (*.json)"
        )
        if not selected:
            return
        try:
            document = capability_workflow_from_json(
                Path(selected).read_text(encoding="utf-8")
            )
            self.controls.set_inputs(capability_workflow_inputs(document))
        except (OSError, TypeError, ValueError) as exc:
            self.status.setText(f"Workflow load failed: {exc}")
        else:
            self.status.setText("Workflow loaded — run when ready.")

    def _export_csv(self) -> None:
        if self._dataset is not None:
            self._save_text(
                "Export Capability Rows",
                "capability-observations.csv",
                scalar_ensemble_csv(self._dataset),
            )

    def _export_json(self) -> None:
        if self._dataset is not None:
            self._save_text(
                "Export Capability Rows",
                "capability-observations.json",
                capability_observation_ensemble_json(self._dataset),
            )

    def _save_text(self, title: str, default_name: str, source: str) -> None:
        selected, _filter = QFileDialog.getSaveFileName(self, title, default_name)
        if not selected:
            return
        try:
            Path(selected).write_text(source + "\n", encoding="utf-8")
        except OSError as exc:
            self.status.setText(f"Export failed: {exc}")
        else:
            self.status.setText(f"Saved {selected}.")


__all__ = ["CapabilityOptimizationTab"]
