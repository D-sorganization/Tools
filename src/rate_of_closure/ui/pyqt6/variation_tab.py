"""Monte-Carlo dispersion and sensitivity UI for the shared variation engine."""

from __future__ import annotations

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
from rate_of_closure.ui.pyqt6.variation_tab_editors import VariationTabEditorsMixin
from rate_of_closure.ui.pyqt6.variation_tab_io import VariationTabIoMixin
from rate_of_closure.ui.pyqt6.variation_tab_results import VariationTabResultsMixin
from rate_of_closure.ui.pyqt6.variation_tab_run import VariationTabRunMixin
from rate_of_closure.ui.pyqt6.variation_worker import VariationWorker
from rate_of_closure.variation.simulation_types import SimulationEnsembleResult
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.variation import (
    CATEGORY_DELIVERY,
    MODES,
    SensitivityResult,
    VariationDataset,
    VariationPlan,
    keys_for_mode,
)

__all__ = ["QFileDialog", "VariationTab"]


class VariationTab(
    VariationTabRunMixin,
    VariationTabEditorsMixin,
    VariationTabIoMixin,
    VariationTabResultsMixin,
    QWidget,
):
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
        self._pending_ensemble_result: SimulationEnsembleResult | None = None
        self._accepted_authority_identity: object | None = None
        self._accepted_result_views = None
        self._accepted_plot_dataset = None
        self._active_authority_identity: object | None = None
        self._active_plan: VariationPlan | None = None
        self._active_compute_sensitivity = False
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
        self._base_combo.currentIndexChanged.connect(self._invalidate_current_study)
        form.addRow("Base Scenario", self._base_combo)

        self._flight_combo = QComboBox()
        self._flight_combo.addItems([m.value for m in FlightModelType])
        self._flight_combo.setCurrentText("waterloo_penner")
        self._flight_combo.setToolTip(
            "Ball-flight model used for every run (kept on the plan so a "
            "saved study replays identically)."
        )
        self._flight_combo.currentIndexChanged.connect(self._invalidate_current_study)
        form.addRow("Flight Model", self._flight_combo)

        self._runs_spin = QSpinBox()
        self._runs_spin.setRange(2, variation_constants.MAX_RUNS)
        self._runs_spin.setValue(200)
        self._runs_spin.setToolTip(
            "Monte-Carlo runs per study. 100-500 resolves dispersion "
            "well; the sensitivity pass repeats this count once per "
            "noise row."
        )
        self._runs_spin.valueChanged.connect(self._invalidate_current_study)
        form.addRow("Runs", self._runs_spin)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999_999)
        self._seed_spin.setValue(0)
        self._seed_spin.setToolTip(
            "Master RNG seed — the same plan and seed always reproduce "
            "the exact same dataset (per-variable seeded streams)."
        )
        self._seed_spin.valueChanged.connect(self._invalidate_current_study)
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
        self._sens_check.toggled.connect(self._invalidate_current_study)
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
        if scenario != self._scenario and self._base_combo.currentIndex() == 1:
            self._invalidate_current_study()
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
        self._refresh_row_contexts()
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
            groups=self._loaded_groups,
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
