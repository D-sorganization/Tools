"""Monte-Carlo dispersion and sensitivity UI for the shared variation engine."""

from __future__ import annotations

import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtWidgets import QFileDialog as QFileDialog

from rate_of_closure.club import get_club
from rate_of_closure.model import MPH_PER_MPS, ImpactScenario
from rate_of_closure.simulation import SimulationConfig
from rate_of_closure.ui.pyqt6 import variation_constants
from rate_of_closure.ui.pyqt6.variation_rows import NoiseRow
from rate_of_closure.ui.pyqt6.variation_tab_io import VariationTabIoMixin
from rate_of_closure.ui.pyqt6.variation_tab_results import (
    VariationTabResultsMixin,
    populate_result_views,
)
from rate_of_closure.ui.pyqt6.variation_worker import VariationWorker
from rate_of_closure.variation.plot_data import build_ensemble_plot_dataset
from rate_of_closure.variation.simulation_types import SimulationEnsembleResult
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    MODES,
    SensitivityResult,
    VariationDataset,
    VariationPlan,
    keys_for_mode,
)

logger = logging.getLogger(__name__)

__all__ = ["QFileDialog", "VariationTab"]


class VariationTab(VariationTabIoMixin, VariationTabResultsMixin, QWidget):
    """Monte-Carlo variation tab (controls left, results right)."""

    #: Emitted after a successful study for landing-scatter overlays (#4125 H7b).
    studyCompleted = pyqtSignal(object)  # noqa: N815 — Qt convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scenario = ImpactScenario(clubhead_speed_mph=113.0)
        self._loaded_base: dict[str, float] = {}
        self._worker: VariationWorker | None = None
        self._generation = 0
        self._simulation_config_valid = True
        self._dataset: VariationDataset | None = None
        self._sensitivity: SensitivityResult | None = None
        self._ensemble_result: SimulationEnsembleResult | None = None
        self._base_simulation_config = SimulationConfig(
            scenario=self._scenario,
            club=get_club("Driver 10.5°"),
            source_kind="double_pendulum",
        )
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

    def _build_setup_box(self) -> QGroupBox:
        box = QGroupBox("Study Setup")
        form = QFormLayout(box)
        self._mode_combo = QComboBox()
        for mode in MODES:
            self._mode_combo.addItem(variation_constants.MODE_LABELS[mode], mode)
        self._mode_combo.setToolTip(
            "Which pipeline slice each run exercises: full delivery → "
            "impact → flight, the double-pendulum swing source feeding it, "
            "or direct launch conditions into ball flight only."
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Pipeline", self._mode_combo)

        self._base_combo = QComboBox()
        self._base_combo.addItems(list(variation_constants.BASE_SOURCES))
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
        self._runs_spin.setRange(2, variation_constants.MAX_RUNS)
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

    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Adopt the explorer's scenario (base-source 'Explorer Scenario')."""
        self._scenario = scenario

    def set_simulation_config(self, config: SimulationConfig) -> None:
        """Set the complete base request used by trace-capable swing studies."""
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")
        changed = config != self._base_simulation_config
        was_valid = self._simulation_config_valid
        was_running = bool(self._worker and self._worker.isRunning())
        if changed:
            self._invalidate_current_study()
        self._base_simulation_config = config
        self._simulation_config_valid = True
        self._set_running(bool(self._worker and self._worker.isRunning()))
        if changed or not was_valid:
            self._status.setText(
                "Simulation changed; cancelling the prior variation study."
                if was_running
                else "Ready with the current Simulation inputs."
            )

    def set_simulation_unavailable(self, message: str) -> None:
        """Fail closed while the Simulation editor has no valid request."""
        self._invalidate_current_study()
        self._simulation_config_valid = False
        self._set_running(bool(self._worker and self._worker.isRunning()))
        self._status.setText(message)

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

    def ensemble_result(self) -> SimulationEnsembleResult | None:
        """Return the most recent complete trace ensemble, when requested."""
        return self._ensemble_result

    def stop(self) -> None:
        """Cancel and join any running worker (window close and tests)."""
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(10_000)

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
        if not self._simulation_config_valid:
            self._status.setText(
                "Cannot run: current Simulation inputs are incomplete or invalid."
            )
            return
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            plan = self.build_plan()
        except (ContractViolationError, ValueError) as exc:
            self._status.setText(f"Cannot run: {exc}")
            return
        self._dataset = None
        self._sensitivity = None
        self._ensemble_result = None
        self._generation += 1
        generation = self._generation
        self._set_running(True)
        worker = VariationWorker(
            plan,
            compute_sensitivity=self._sens_check.isChecked(),
            base_simulation_config=self._base_simulation_config,
        )
        worker.progressed.connect(
            lambda report, current=generation: self._accept_progress(current, report)
        )
        worker.phaseChanged.connect(
            lambda phase, current=generation: self._accept_phase(current, phase)
        )
        worker.succeeded.connect(
            lambda dataset, sensitivity, current=generation: self._accept_succeeded(
                current, dataset, sensitivity
            )
        )
        worker.ensembleSucceeded.connect(
            lambda result, current=generation: self._accept_ensemble_succeeded(
                current, result
            )
        )
        worker.cancelled.connect(
            lambda current=generation: self._accept_cancelled(current)
        )
        worker.failed.connect(
            lambda message, current=generation: self._accept_failed(current, message)
        )
        worker.finished.connect(
            lambda current=generation, owner=worker: self._accept_finished(
                current, owner
            )
        )
        self._worker = worker
        self._progress.setRange(0, worker.total_runs)
        self._progress.setValue(0)
        self._status.setText("Running…")
        worker.start()

    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running and self._simulation_config_valid)
        self._cancel_button.setEnabled(running)
        self._export_csv.setEnabled(not running and self._dataset is not None)
        self._export_json.setEnabled(not running and self._dataset is not None)
        has_ensemble = self._ensemble_result is not None
        self._export_trace_csv.setEnabled(not running and has_ensemble)
        self._export_ensemble_json.setEnabled(not running and has_ensemble)

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status.setText("Cancelling…")

    # ── worker callbacks (GUI thread) ───────────────────────────────
    def _is_current_generation(self, generation: int) -> bool:
        return generation == self._generation

    def _accept_progress(self, generation: int, report: object) -> None:
        if self._is_current_generation(generation):
            self._on_progress(report)

    def _accept_phase(self, generation: int, phase: str) -> None:
        if self._is_current_generation(generation):
            self._on_phase(phase)

    def _accept_succeeded(
        self, generation: int, dataset: object, sensitivity: object
    ) -> None:
        if self._is_current_generation(generation) and isinstance(
            dataset, VariationDataset
        ):
            self._on_succeeded(dataset, sensitivity)

    def _accept_ensemble_succeeded(self, generation: int, result: object) -> None:
        if self._is_current_generation(generation) and isinstance(
            result, SimulationEnsembleResult
        ):
            self._on_ensemble_succeeded(result)

    def _accept_cancelled(self, generation: int) -> None:
        if self._is_current_generation(generation):
            self._on_cancelled()

    def _accept_failed(self, generation: int, message: str) -> None:
        if self._is_current_generation(generation):
            self._on_failed(message)

    def _accept_finished(self, generation: int, worker: VariationWorker) -> None:
        owns_current_slot = worker is self._worker
        if owns_current_slot:
            self._worker = None
        if self._is_current_generation(generation):
            self._on_finished()
        elif owns_current_slot:
            self._set_running(False)

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
        if self._ensemble_result is None:
            self._ensemble_scatter.set_variation_dataset(dataset)
            self._distribution_matrix.set_variation_dataset(dataset)
        self._populate_results()
        failures = dataset.plan.n_runs - dataset.n_success
        note = f" ({failures} runs failed)" if failures else ""
        self._status.setText(
            f"Done: {dataset.n_success}/{dataset.plan.n_runs} runs in "
            f"{dataset.elapsed_s:.1f} s{note}."
        )
        self.studyCompleted.emit(dataset)

    def _on_ensemble_succeeded(self, result: SimulationEnsembleResult) -> None:
        """Populate complete-trace views before the scalar completion callback."""
        self._ensemble_result = result
        self._landing.set_outcomes(tuple(outcome.status for outcome in result.outcomes))
        self._export_trace_csv.setEnabled(True)
        self._export_ensemble_json.setEnabled(True)
        plot_dataset = build_ensemble_plot_dataset(result)
        self._ensemble_scatter.set_plot_dataset(plot_dataset)
        self._distribution_matrix.set_plot_dataset(plot_dataset)
        self._arc_overlay.set_plot_dataset(plot_dataset)

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
        populate_result_views(
            dataset,
            self._sensitivity,
            self._summary_table,
            self._sensitivity_table,
            self._spearman_table,
            self._landing,
        )

    def _invalidate_current_study(self) -> None:
        self._generation += 1
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        self._dataset = None
        self._sensitivity = None
        self._ensemble_result = None
        self._summary_table.setRowCount(0)
        for table in (self._sensitivity_table, self._spearman_table):
            table.setRowCount(0)
            table.setColumnCount(0)
        self._landing.clear_view()
        self._ensemble_scatter.clear_view()
        self._distribution_matrix.clear_view()
        self._arc_overlay.clear_view()

    # ── export / plan IO ────────────────────────────────────────────
