"""Monte-Carlo dispersion and sensitivity UI for the shared variation engine."""

from __future__ import annotations

import json
import logging

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.model import MPH_PER_MPS, ImpactScenario
from rate_of_closure.ui.pyqt6.variation_results import (
    LandingCanvas,
    SensitivityTable,
    SummaryTable,
)
from rate_of_closure.ui.pyqt6.variation_rows import NoiseRow
from rate_of_closure.ui.pyqt6.variation_worker import VariationWorker
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    MODES,
    SensitivityResult,
    VariationDataset,
    VariationPlan,
    dispersion_ellipse,
    keys_for_mode,
    spearman_matrix,
    summary_stats,
)
from shared.python.swing_sim.variation.dataset_io import write_csv, write_json

logger = logging.getLogger(__name__)

__all__ = ["VariationTab"]

_MODE_LABELS: dict[str, str] = {
    "delivery": "Delivery → Impact → Flight",
    "swing": "Pendulum Swing → Impact → Flight",
    "launch": "Launch Conditions → Flight",
}
_BASE_SOURCES = ("Registry Defaults", "Explorer Scenario")
_MAX_RUNS = 5000


class VariationTab(QWidget):
    """Monte-Carlo variation tab (controls left, results right)."""

    #: Emitted with the VariationDataset after a successful study
    #: (#4125 H7b: the course view overlays the landing scatter).
    studyCompleted = pyqtSignal(object)  # noqa: N815 — Qt convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scenario = ImpactScenario(clubhead_speed_mph=113.0)
        self._loaded_base: dict[str, float] = {}
        self._worker: VariationWorker | None = None
        self._dataset: VariationDataset | None = None
        self._sensitivity: SensitivityResult | None = None
        self._rows: list[NoiseRow] = []

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_setup_box())
        left_layout.addWidget(self._build_noise_box())
        left_layout.addWidget(self._build_run_box())
        left_layout.addWidget(self._build_export_box())
        left_layout.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(left)
        scroll.setMinimumWidth(430)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_results_tabs())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._add_row()

    # ── construction ────────────────────────────────────────────────
    def _build_setup_box(self) -> QGroupBox:
        box = QGroupBox("Study Setup")
        form = QFormLayout(box)
        self._mode_combo = QComboBox()
        for mode in MODES:
            self._mode_combo.addItem(_MODE_LABELS[mode], mode)
        self._mode_combo.setToolTip(
            "Which pipeline slice each run exercises: full delivery → "
            "impact → flight, the double-pendulum swing source feeding it, "
            "or direct launch conditions into ball flight only."
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Pipeline", self._mode_combo)

        self._base_combo = QComboBox()
        self._base_combo.addItems(list(_BASE_SOURCES))
        self._base_combo.setToolTip(
            "Base values the noise varies about: the shared registry "
            "defaults, or the current explorer scenario (clubhead speed "
            "and impact offsets carried over in delivery mode)."
        )
        form.addRow("Base Scenario", self._base_combo)

        self._flight_combo = QComboBox()
        self._flight_combo.addItems([m.value for m in FlightModelType])
        self._flight_combo.setCurrentText("waterloo_penner")
        self._flight_combo.setToolTip(
            "Ball-flight model used for every run (kept on the plan so a "
            "saved study replays identically)."
        )
        form.addRow("Flight Model", self._flight_combo)

        self._runs_spin = QSpinBox()
        self._runs_spin.setRange(2, _MAX_RUNS)
        self._runs_spin.setValue(200)
        self._runs_spin.setToolTip(
            "Monte-Carlo runs per study. 100-500 resolves dispersion "
            "well; the sensitivity pass repeats this count once per "
            "noise row."
        )
        form.addRow("Runs", self._runs_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999_999)
        self._seed_spin.setValue(0)
        self._seed_spin.setToolTip(
            "Master RNG seed — the same plan and seed always reproduce "
            "the exact same dataset (per-variable seeded streams)."
        )
        form.addRow("Seed", self._seed_spin)
        return box

    def _build_noise_box(self) -> QGroupBox:
        box = QGroupBox("Varied Variables (Noise)")
        self._rows_layout = QVBoxLayout(box)
        self._rows_layout.setSpacing(4)
        header = QLabel("Variable · distribution · scale · optional clipping")
        header.setToolTip(
            "Each row adds run-to-run noise to one registry variable; "
            "hover any editor for its unit-aware guidance."
        )
        self._rows_layout.addWidget(header)
        add = QPushButton("Add Variable")
        add.setToolTip("Add another noise row (one per variable).")
        add.clicked.connect(self._add_row)
        self._rows_layout.addWidget(add)
        return box

    def _build_run_box(self) -> QGroupBox:
        box = QGroupBox("Run")
        layout = QVBoxLayout(box)
        self._sens_check = QCheckBox("Compute One-at-a-Time Sensitivity")
        self._sens_check.setChecked(True)
        self._sens_check.setToolTip(
            "After the main batch, rerun the study once per noise row with "
            "only that row active (paired draws) to attribute each "
            "output's spread to its inputs. Multiplies runtime by the "
            "number of rows + 1."
        )
        layout.addWidget(self._sens_check)
        row = QHBoxLayout()
        self._run_button = QPushButton("Run Variation Study")
        self._run_button.setToolTip(
            "Sample every noise row, run the pipeline once per run on a "
            "worker thread, and populate the results tabs."
        )
        self._run_button.clicked.connect(self._on_run)
        row.addWidget(self._run_button, stretch=1)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setEnabled(False)
        self._cancel_button.setToolTip(
            "Cooperatively cancel the running study; in-flight runs stop."
        )
        self._cancel_button.clicked.connect(self._on_cancel)
        row.addWidget(self._cancel_button)
        layout.addLayout(row)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        layout.addWidget(self._progress)
        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        return box

    def _build_export_box(self) -> QGroupBox:
        box = QGroupBox("Export / Reproduce")
        layout = QHBoxLayout(box)
        self._export_csv = QPushButton("Dataset CSV")
        self._export_csv.setToolTip(
            "Export the runs table (inputs, outputs, success flags) as CSV."
        )
        self._export_csv.clicked.connect(self._on_export_csv)
        self._export_json = QPushButton("Dataset JSON")
        self._export_json.setToolTip(
            "Export the full dataset + plan as JSON (documented, re-importable)."
        )
        self._export_json.clicked.connect(self._on_export_json)
        save_plan = QPushButton("Save Plan")
        save_plan.setToolTip(
            "Save just the plan as JSON — the schema the web tab also reads."
        )
        save_plan.clicked.connect(self._on_save_plan)
        load_plan = QPushButton("Load Plan")
        load_plan.setToolTip("Load a plan JSON back into the editors.")
        load_plan.clicked.connect(self._on_load_plan)
        for button in (self._export_csv, self._export_json):
            button.setEnabled(False)
        for button in (self._export_csv, self._export_json, save_plan, load_plan):
            layout.addWidget(button)
        return box

    def _build_results_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        self._summary_table = SummaryTable()
        self._sensitivity_table = SensitivityTable()
        self._sensitivity_table.setToolTip(
            "One-at-a-time sensitivity: std induced in each output "
            "(column) when only that input's (row) noise is active. "
            "Hot cells dominate their column."
        )
        self._spearman_table = SensitivityTable()
        self._spearman_table.setToolTip(
            "Spearman rank correlation between each sampled input and "
            "each output over the full study — a cheap global-sensitivity "
            "cross-check of the one-at-a-time matrix."
        )
        self._landing = LandingCanvas()
        tabs.addTab(self._summary_table, "Summary")
        tabs.addTab(self._sensitivity_table, "Sensitivity")
        tabs.addTab(self._spearman_table, "Rank Correlation")
        tabs.addTab(self._landing, "Landing Dispersion")
        return tabs

    # ── public API ──────────────────────────────────────────────────
    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Adopt the explorer's scenario (base-source 'Explorer Scenario')."""
        self._scenario = scenario

    def mode(self) -> str:
        """The selected pipeline mode."""
        return str(self._mode_combo.currentData())

    def build_plan(self) -> VariationPlan:
        """The VariationPlan described by the editors (DbC-validated)."""
        mode = self.mode()
        legal = set(keys_for_mode(mode))
        base: dict[str, float] = {
            key: value for key, value in self._loaded_base.items() if key in legal
        }
        if self._base_combo.currentIndex() == 1 and mode in ("delivery", "swing"):
            s = self._scenario
            if mode == "delivery":
                key = f"{CATEGORY_DELIVERY}.clubhead_speed_mps"
                base[key] = s.clubhead_speed_mph / MPH_PER_MPS
            base[f"{CATEGORY_DELIVERY}.impact_offset_toe_mm"] = s.impact_offset_toe_mm
            base[f"{CATEGORY_DELIVERY}.impact_offset_high_mm"] = s.impact_offset_high_mm
        return VariationPlan(
            mode=mode,
            base_variables=base,
            noise=tuple(row.to_spec() for row in self._rows),
            n_runs=self._runs_spin.value(),
            seed=self._seed_spin.value(),
            flight_model=self._flight_combo.currentText(),
        )

    def dataset(self) -> VariationDataset | None:
        """The most recent completed dataset, if any."""
        return self._dataset

    def stop(self) -> None:
        """Cancel and join any running worker (window close and tests)."""
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(10_000)

    # ── noise rows ──────────────────────────────────────────────────
    def _add_row(self) -> NoiseRow:
        row = NoiseRow(self.mode(), self._remove_row)
        self._rows.append(row)
        # Insert above the trailing "Add Variable" button.
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        return row

    def _remove_row(self, row: NoiseRow) -> None:
        if len(self._rows) <= 1:
            self._status.setText("At least one noise row is required.")
            return
        self._rows.remove(row)
        row.setParent(None)
        row.deleteLater()

    def _on_mode_changed(self, *_args: object) -> None:
        self._loaded_base.clear()
        for row in self._rows:
            row.set_mode(self.mode())

    # ── run / cancel ────────────────────────────────────────────────
    def _on_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            plan = self.build_plan()
        except (ContractViolationError, ValueError) as exc:
            self._status.setText(f"Cannot run: {exc}")
            return
        self._dataset = None
        self._sensitivity = None
        self._set_running(True)
        worker = VariationWorker(plan, compute_sensitivity=self._sens_check.isChecked())
        worker.progressed.connect(self._on_progress)
        worker.phaseChanged.connect(self._on_phase)
        worker.succeeded.connect(self._on_succeeded)
        worker.cancelled.connect(self._on_cancelled)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self._progress.setRange(0, worker.total_runs)
        self._progress.setValue(0)
        self._status.setText("Running…")
        worker.start()

    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._export_csv.setEnabled(not running and self._dataset is not None)
        self._export_json.setEnabled(not running and self._dataset is not None)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling…")

    # ── worker callbacks (GUI thread) ───────────────────────────────
    def _on_progress(self, report: object) -> None:
        iteration = int(getattr(report, "iteration", 0))
        failed = int(getattr(report, "cost", 0.0))
        self._progress.setValue(min(iteration, self._progress.maximum()))
        note = f", {failed} failed" if failed else ""
        self._status.setText(
            f"Run {iteration}/{self._progress.maximum()}{note} — "
            f"{getattr(report, 'elapsed_s', 0.0):.1f} s"
        )

    def _on_phase(self, phase: str) -> None:
        if phase.startswith("Sensitivity"):
            self._progress.setRange(0, 0)  # busy while OAT sub-studies run
        self._status.setText(phase)

    def _on_succeeded(self, dataset: VariationDataset, sensitivity: object) -> None:
        self._dataset = dataset
        self._sensitivity = (
            sensitivity if isinstance(sensitivity, SensitivityResult) else None
        )
        self._populate_results()
        failures = dataset.plan.n_runs - dataset.n_success
        note = f" ({failures} runs failed)" if failures else ""
        self._status.setText(
            f"Done: {dataset.n_success}/{dataset.plan.n_runs} runs in "
            f"{dataset.elapsed_s:.1f} s{note}."
        )
        self.studyCompleted.emit(dataset)

    def _on_cancelled(self) -> None:
        self._status.setText("Cancelled.")

    def _on_failed(self, message: str) -> None:
        self._status.setText(f"Study failed: {message}")

    def _on_finished(self) -> None:
        self._progress.setRange(0, max(self._progress.maximum(), 1))
        self._progress.setValue(self._progress.maximum())
        self._set_running(False)

    # ── results ─────────────────────────────────────────────────────
    def _populate_results(self) -> None:
        dataset = self._dataset
        if dataset is None:
            return
        self._summary_table.set_stats(summary_stats(dataset))
        rho = spearman_matrix(dataset)
        self._spearman_table.set_matrix(
            dataset.input_names,
            dataset.output_names,
            rho,
            np.abs(rho),
            value_format="{:+.2f}",
        )
        if self._sensitivity is not None:
            self._sensitivity_table.set_matrix(
                self._sensitivity.input_keys,
                self._sensitivity.output_names,
                self._sensitivity.matrix,
                self._sensitivity.normalized,
            )
        try:
            ellipse = dispersion_ellipse(dataset)
        except (ContractViolationError, ValueError):
            ellipse = None
        self._landing.set_dataset(dataset, ellipse)

    # ── export / plan IO ────────────────────────────────────────────
    def _on_export_csv(self) -> None:
        if self._dataset is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Dataset CSV", "variation_dataset.csv", "CSV (*.csv)"
        )
        if path:
            write_csv(self._dataset, path)
            self._status.setText(f"Dataset CSV written to {path}.")

    def _on_export_json(self) -> None:
        if self._dataset is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export Dataset JSON", "variation_dataset.json", "JSON (*.json)"
        )
        if path:
            write_json(self._dataset, path)
            self._status.setText(f"Dataset JSON written to {path}.")

    def _on_save_plan(self) -> None:
        try:
            plan = self.build_plan()
        except (ContractViolationError, ValueError) as exc:
            self._status.setText(f"Cannot save plan: {exc}")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save Variation Plan", "variation_plan.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(plan.dumps())
            self._status.setText(f"Plan saved to {path}.")

    def _on_load_plan(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load Variation Plan", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                plan = VariationPlan.loads(handle.read())
        except (ContractViolationError, ValueError, json.JSONDecodeError) as exc:
            self._status.setText(f"Cannot load plan: {exc}")
            return
        self.load_plan(plan)
        self._status.setText(f"Plan loaded from {path}.")

    def load_plan(self, plan: VariationPlan) -> None:
        """Drive all editors from a plan (used by Load Plan and tests)."""
        self._mode_combo.setCurrentIndex(MODES.index(plan.mode))
        self._runs_spin.setValue(plan.n_runs)
        self._seed_spin.setValue(plan.seed)
        self._flight_combo.setCurrentText(plan.flight_model)
        self._base_combo.setCurrentIndex(0)  # the plan's own base values win
        self._loaded_base = dict(plan.base_variables)
        while len(self._rows) > 1:
            self._remove_row(self._rows[-1])
        for i, spec in enumerate(plan.noise):
            row = self._rows[0] if i == 0 else self._add_row()
            row.load_spec(spec)
