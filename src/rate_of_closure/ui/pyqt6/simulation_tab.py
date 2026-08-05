"""Simulation session UI for source setup, playback, results, and inspection."""

from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.club import club_names, get_club
from rate_of_closure.derivation import LAUNCH_EXPLANATIONS
from rate_of_closure.derivation_models import DerivationConfig
from rate_of_closure.model import MPH_PER_MPS, ImpactScenario
from rate_of_closure.simulation import (
    SOURCE_KINDS,
    BallSetup,
    ContactMode,
    SimulationConfig,
    SimulationRun,
    delivery_at,
    make_source,
    run_simulation,
)
from rate_of_closure.simulation.targets import TargetRegion, layout_for_region
from rate_of_closure.ui.pyqt6.ball_setup_control import BallSetupControl
from rate_of_closure.ui.pyqt6.flight_view import FlightView
from rate_of_closure.ui.pyqt6.inspector_view import InspectorView
from rate_of_closure.ui.pyqt6.kinetics_panel import KineticsPanel
from rate_of_closure.ui.pyqt6.result_row import ResultRow, explanation_html
from rate_of_closure.ui.pyqt6.simulation_specs import (
    LAUNCH_ROWS,
    SOURCE_LABELS,
    TILT_SPECS,
)
from rate_of_closure.ui.pyqt6.simulation_view import SimulationView
from rate_of_closure.ui.pyqt6.solver_panel import SolverPanel
from rate_of_closure.ui.pyqt6.strike_view import StrikeView
from rate_of_closure.ui.pyqt6.torque_profile_controller import RunMode
from rate_of_closure.ui.pyqt6.torque_profile_panel import TorqueProfilePanel
from rate_of_closure.units import FIELD_GUIDANCE, format_distance_m
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.run_config import DoublePendulumRunConfig
from shared.python.swing_sim.types import PlaneOrientation

logger = logging.getLogger(__name__)

__all__ = ["LAUNCH_ROWS", "SOURCE_LABELS", "SimulationTab"]

_SCRUB_STEPS = 1000


class SimulationTab(QWidget):
    """Simulation session tab (controls left, scene/inspector right)."""

    #: Emitted with the SimulationRun after every successful run.
    runCompleted = pyqtSignal(object)  # noqa: N815 - Qt signal convention
    #: Emitted with a glossary term key when an explanation link is used.
    glossaryRequested = pyqtSignal(str)  # noqa: N815 - Qt signal convention
    #: Drives conditional Calculation Description sections from model changes.
    configChanged = pyqtSignal(object)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scenario = ImpactScenario(clubhead_speed_mph=113.0)
        self._run: SimulationRun | None = None
        self._tau: float | None = None  # None = auto (max clubhead speed)
        self._source = None  # cached app-frame source for live scrubbing
        self._rows: dict[str, ResultRow] = {}

        self._view = SimulationView()
        self._strike_view = StrikeView()
        self._flight_view = FlightView()
        self._kinetics_panel = KineticsPanel()
        self._kinetics_panel.glossaryRequested.connect(self.glossaryRequested)
        self._inspector = InspectorView()
        self._solver_panel = SolverPanel()
        self._torque_profile_panel = TorqueProfilePanel()
        self._torque_profile_panel.runModeChanged.connect(
            self._on_torque_selection_changed
        )
        self._torque_profile_panel.profileChanged.connect(
            self._on_torque_selection_changed
        )
        self._torque_profile_panel.jointLocksChanged.connect(
            self._on_joint_locks_changed
        )
        self._torque_profile_panel.fitCurrentRunRequested.connect(self._fit_current_run)
        self._solver_panel.applyRequested.connect(self.apply_solver_solution)
        # Keep the course scene and flight overlay aligned with target edits.
        self._solver_panel.target_panel().regionChanged.connect(
            self._on_target_region_changed
        )

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(self._build_setup_box())
        self._scrub_box = self._build_scrub_box()
        left_layout.addWidget(self._scrub_box)
        left_layout.addWidget(self._build_launch_box())
        left_layout.addWidget(self._build_explanation_box())
        left_layout.addStretch(1)
        # Scrolling preserves readable entries in small windows.
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.Shape.NoFrame)
        left.setWidget(left_content)
        left.setMinimumWidth(300)

        # Scale-separated face, swing, kinetics, and flight displays.
        right = QTabWidget()
        right.addTab(self._strike_view, "Strike")
        right.addTab(self._view, "Swing")
        right.addTab(self._kinetics_panel, "Kinetics")
        right.addTab(self._flight_view, "Flight")
        right.addTab(self._inspector, "Inspector")
        right.addTab(self._solver_panel, "Solver")
        right.addTab(self._torque_profile_panel, "Torque Profiles")
        right.setCurrentWidget(self._view)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._show_explanation(LAUNCH_ROWS[0][0])

    def _build_setup_box(self) -> QGroupBox:
        box = QGroupBox("Simulation Setup")
        form = QFormLayout(box)

        self._source_combo = QComboBox()
        self._source_combo.addItems([SOURCE_LABELS[kind] for kind in SOURCE_KINDS])
        self._source_combo.setToolTip(FIELD_GUIDANCE["swing_source"])
        self._source_combo.currentIndexChanged.connect(self._invalidate_source)
        self._source_combo.currentIndexChanged.connect(
            self._reconcile_joint_locks_for_source
        )
        form.addRow("Swing Source", self._source_combo)
        self._tilt_spins: dict[str, QDoubleSpinBox] = {}
        for attr, label, guidance_key in TILT_SPECS:
            spin = QDoubleSpinBox()
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setKeyboardTracking(False)
            spin.setDecimals(1)
            spin.setRange(-90.0, 90.0)
            spin.setSuffix(" deg")
            spin.setMinimumWidth(84)  # stays readable at small windows
            spin.setToolTip(FIELD_GUIDANCE[guidance_key])
            spin.valueChanged.connect(self._invalidate_source)
            self._tilt_spins[attr] = spin
            form.addRow(label, spin)
        self._tilt_spins["side_tilt_deg"].setValue(-45.0)
        self._club_combo = QComboBox()
        self._club_combo.addItems(club_names())
        self._club_combo.setCurrentText("Driver 10.5°")
        self._club_combo.setToolTip(FIELD_GUIDANCE["club_selection"])
        self._club_combo.currentTextChanged.connect(self._on_club_changed)
        form.addRow("Club", self._club_combo)

        club = get_club(self._club_combo.currentText())
        default_setup = SimulationConfig(scenario=self._scenario, club=club).ball_setup
        self._ball_setup_control = BallSetupControl(default_setup, club.name)
        self._ball_setup_control.setupChanged.connect(self._emit_config)
        form.addRow(self._ball_setup_control)

        self._contact_combo = QComboBox()
        self._contact_combo.addItem(
            "Delivery Inspection (Forced Alignment)",
            ContactMode.DELIVERY_INSPECTION,
        )
        self._contact_combo.addItem(
            "Fixed Ball Contact (Detect Hit / Miss)",
            ContactMode.FIXED_BALL_CONTACT,
        )
        self._contact_combo.setToolTip(
            "Choose forced delivery inspection or sampled fixed-ball contact. "
            "Suggested use: inspection for delivery studies; fixed-ball contact "
            "for honest hit/miss evaluation. Source: Rate of Closure contact "
            "contract; sampled contact is a point-to-sphere approximation."
        )
        self._contact_combo.currentIndexChanged.connect(self._on_contact_mode_changed)
        form.addRow("Contact Policy", self._contact_combo)
        self._contact_description = QLabel()
        self._contact_description.setWordWrap(True)
        form.addRow(self._contact_description)

        self._flight_combo = QComboBox()
        self._flight_combo.addItems([m.value for m in FlightModelType])
        self._flight_combo.setCurrentText("waterloo_penner")
        self._flight_combo.setToolTip(FIELD_GUIDANCE["flight_model"])
        self._flight_combo.currentIndexChanged.connect(self._emit_config)
        self._source_combo.currentIndexChanged.connect(self._emit_config)
        for spin in self._tilt_spins.values():
            spin.valueChanged.connect(self._emit_config)
        form.addRow("Flight Model", self._flight_combo)

        self._run_button = QPushButton("Run Simulation")
        self._run_button.setToolTip(
            "Generate the swing, solve the impact at the scrubbed instant, "
            "and integrate the ball flight."
        )
        self._run_button.clicked.connect(self.run_now)
        form.addRow(self._run_button)
        self._run_status = QLabel(
            "Stale — Run Simulation to calculate the current configuration."
        )
        self._run_status.setWordWrap(True)
        self._run_status.setFrameShape(QFrame.Shape.StyledPanel)
        self._run_status.setMargin(8)
        font = self._run_status.font()
        font.setBold(True)
        self._run_status.setFont(font)
        self._run_status.setAccessibleName("Simulation Run Status")
        form.addRow(self._run_status)
        self._update_contact_controls()
        return box

    def _build_scrub_box(self) -> QGroupBox:
        box = QGroupBox("Impact Time (Scrub the Swing Onto the Ball)")
        layout = QVBoxLayout(box)

        row = QHBoxLayout()
        self._scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self._scrub_slider.setRange(0, _SCRUB_STEPS)
        self._scrub_slider.setValue(_SCRUB_STEPS // 2)
        self._scrub_slider.setToolTip(FIELD_GUIDANCE["impact_time_scrub"])
        self._scrub_slider.valueChanged.connect(self._on_scrub_moved)
        self._scrub_slider.sliderReleased.connect(self._on_scrub_released)
        row.addWidget(self._scrub_slider, stretch=1)
        self._scrub_label = QLabel("auto")
        self._scrub_label.setFixedWidth(72)
        row.addWidget(self._scrub_label)
        layout.addLayout(row)

        self._auto_tau_button = QPushButton("Auto (Max Clubhead Speed)")
        self._auto_tau_button.setToolTip(
            "Reset the impact instant to the sampled moment of maximum clubhead speed."
        )
        self._auto_tau_button.clicked.connect(self._on_auto_tau)
        layout.addWidget(self._auto_tau_button)

        self._delivery_label = QLabel("—")
        self._delivery_label.setWordWrap(True)
        layout.addWidget(self._delivery_label)
        return box

    def _build_launch_box(self) -> QGroupBox:
        box = QGroupBox("Launch Numbers")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        for field, label, _unit in LAUNCH_ROWS:
            row = ResultRow(field, label)
            row.clicked.connect(self._show_explanation)
            self._rows[field] = row
            layout.addWidget(row)
        return box

    def _build_explanation_box(self) -> QGroupBox:
        box = QGroupBox("What This Number Means")
        layout = QVBoxLayout(box)
        self._explanation = QTextBrowser()
        self._explanation.setOpenExternalLinks(False)
        self._explanation.setOpenLinks(False)
        self._explanation.setToolTip(
            "Explanation of the selected row; the Glossary link jumps "
            "to the matching term."
        )
        self._explanation.anchorClicked.connect(self._on_explanation_link)
        self._explanation.setMinimumHeight(90)
        self._explanation.setMaximumHeight(150)
        layout.addWidget(self._explanation)
        return box

    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Adopt the explorer's scenario (drives the manual source)."""
        self._scenario = scenario
        self._invalidate_source()

    def plane(self) -> PlaneOrientation:
        """The plane orientation described by the tilt inputs."""
        return PlaneOrientation(
            yaw_deg=self._tilt_spins["yaw_deg"].value(),
            side_tilt_deg=self._tilt_spins["side_tilt_deg"].value(),
            forward_tilt_deg=self._tilt_spins["forward_tilt_deg"].value(),
        )

    def source_kind(self) -> str:
        """The selected swing-source kind."""
        return str(SOURCE_KINDS[int(self._source_combo.currentIndex())])

    def derivation_config(self) -> DerivationConfig:
        """The DerivationConfig described by the current controls."""
        plane = self.plane()
        return DerivationConfig(
            flight_model=self._flight_combo.currentText(),
            swing_source=self.source_kind(),
            gear_effect=True,  # the session pipeline always applies it
            plane_tilts_deg=(
                plane.yaw_deg,
                plane.side_tilt_deg,
                plane.forward_tilt_deg,
            ),
        )

    def _emit_config(self, *_args: object) -> None:
        self._mark_stale()
        self.configChanged.emit(self.derivation_config())

    def _on_club_changed(self, name: str) -> None:
        """Apply the canonical club default unless the user owns an override."""
        club = get_club(name)
        default_setup = SimulationConfig(scenario=self._scenario, club=club).ball_setup
        self._ball_setup_control.apply_club_default(default_setup, club.name)
        self._emit_config()

    def contact_mode(self) -> ContactMode:
        """The selected contact policy."""
        return self._contact_combo.currentData()

    def config(self) -> SimulationConfig:
        """The simulation request described by the controls."""
        selection = self._torque_profile_panel.selection()
        joint_locks = self._torque_profile_panel.joint_locks()
        run_config = DoublePendulumRunConfig(joint_locks=joint_locks)
        torque_library = None
        source_kind = self.source_kind()
        if selection.mode is RunMode.PRESCRIBED_TORQUE:
            if not selection.execution_ready or selection.profile is None:
                raise ValueError(selection.validation_message)
            source_kind = "double_pendulum"
            run_config = DoublePendulumRunConfig.prescribed(
                selection.profile.profile_id,
                joint_locks=joint_locks,
            )
            torque_library = self._torque_profile_panel.canonical_library()
        return SimulationConfig(
            scenario=self._scenario,
            club=get_club(self._club_combo.currentText()),
            ball_setup=self._ball_setup_control.setup(),
            source_kind=source_kind,
            plane=self.plane(),
            impact_time_s=(
                self._tau
                if self.contact_mode() is ContactMode.DELIVERY_INSPECTION
                else None
            ),
            flight_model=self._flight_combo.currentText(),
            contact_mode=self.contact_mode(),
            swing_run_config=run_config,
            torque_library=torque_library,
        )

    def run_now(self) -> SimulationRun | None:
        """Run the simulation and populate the scene + inspector."""
        try:
            run = run_simulation(self.config())
        except Exception as exc:  # noqa: BLE001 — surface physics failures
            logger.warning("simulation failed: %s", exc)
            self._set_run_status(f"Error — Simulation failed: {exc}", "error")
            return None
        self._run = run
        self._tau = run.impact_time_s
        self._sync_scrub_after_run(run)
        self._view.set_run(run)
        self._strike_view.set_run(run)
        self._kinetics_panel.set_run(run)
        self._flight_view.set_run(run)
        self._inspector.set_run(run)
        self._refresh_launch_rows()
        self._update_outcome_labels(run)
        self._set_completed_status(run)
        if run.config.swing_run_config.prescribed_profile_id is not None:
            self._torque_profile_panel.set_execution_status(
                "Prescribed profile executed in the double-pendulum dynamics kernel; "
                f"{self._torque_profile_panel.joint_lock_summary()}."
            )
        self.runCompleted.emit(run)
        return run

    def _refresh_launch_rows(self) -> None:
        """Format launch rows; carry follows the distance display unit
        (#4125 H6 — yards default; apex stays in metres)."""
        run = self._run
        if run is None:
            return
        if run.launch is None:
            for row in self._rows.values():
                row.value_label.setText("N/A — No Impact")
                row.setToolTip(
                    "No launch value exists because fixed-ball contact was not "
                    "detected."
                )
            return
        for field, _label, unit in LAUNCH_ROWS:
            value = run.launch[field]
            if not math.isfinite(value):
                text = "—"
            elif field == "carry_m":
                text = f"+{format_distance_m(value)}"
            else:
                text = f"{value:+.1f}{unit}"
            self._rows[field].value_label.setText(text)
            self._rows[field].setToolTip(
                "Click for the explanation and derivation trace"
            )

    def refresh_units(self) -> None:
        """Re-render distance surfaces after a display-unit change."""
        self._refresh_launch_rows()
        self._solver_panel.target_panel().refresh_units()
        # Redraw the flight view so its axes pick up the new unit.
        self._flight_view.set_run(self._run)

    def last_run(self) -> SimulationRun | None:
        """The most recent successful run, if any."""
        return self._run

    def ball_setup_control(self) -> BallSetupControl:
        """Return the canonical Ground/Tee editor hosted by this session."""
        return self._ball_setup_control

    def set_ball_setup(self, setup: BallSetup) -> None:
        """Load a canonical persisted setup without introducing a UI schema."""
        self._ball_setup_control.set_setup(setup)
        self._emit_config()

    def view(self) -> SimulationView:
        """The swing-scale 3D scene (playback controls live on it)."""
        return self._view

    def strike_view(self) -> StrikeView:
        """The face-scale impact-zone viewer."""
        return self._strike_view

    def flight_view(self) -> FlightView:
        """The flight-scale trajectory viewer."""
        return self._flight_view

    def kinetics_panel(self) -> KineticsPanel:
        """The kinetics plots + peak-table sub-tab (#4125 H2)."""
        return self._kinetics_panel

    def inspector(self) -> InspectorView:
        """The run-data inspector."""
        return self._inspector

    def solver_panel(self) -> SolverPanel:
        """The goal-driven Solver panel (worker lifecycle lives on it)."""
        return self._solver_panel

    def _on_target_region_changed(self, region: TargetRegion) -> None:
        """H7b: reflect the edited target in the course scene + overlay."""
        layout = layout_for_region(region)
        self._flight_view.set_course_layout(layout)
        self._flight_view.set_target_region(region)
        self._view.set_course_layout(layout)

    def set_landing_scatter(
        self, carries_m: np.ndarray | None, laterals_m: np.ndarray | None = None
    ) -> None:
        """Variation tie-in (#4125 H7b): forward the landing scatter."""
        self._flight_view.set_landing_scatter(carries_m, laterals_m)

    def apply_solver_solution(
        self, result: object, use_swing_source: bool
    ) -> SimulationRun | None:
        """Load a SolverResult's variables into the session and rerun.

        Mapping (documented deviation — the session's delivery convention
        is a square face at the club's loft, so face angle / dynamic loft
        solutions inform the goal table but are not replayed):

        * both modes: the solved impact offsets land in the scenario;
        * delivery mode: the manual constant-twist source is selected and
          the solved clubhead speed becomes the scenario reference speed;
        * swing-source mode: the double-pendulum source is selected, the
          solved plane tilts drive the tilt inputs, and the solved
          impact-time offset shifts tau off the peak-speed instant.
        """
        variables: dict[str, float] = result.variables  # type: ignore[attr-defined]
        updates = {
            "impact_offset_toe_mm": variables["impact_offset_toe_mm"],
            "impact_offset_high_mm": variables["impact_offset_high_mm"],
        }
        if use_swing_source:
            self._source_combo.setCurrentIndex(SOURCE_KINDS.index("double_pendulum"))
            for attr, var in (
                ("yaw_deg", "swing_yaw_deg"),
                ("side_tilt_deg", "swing_side_tilt_deg"),
                ("forward_tilt_deg", "swing_forward_tilt_deg"),
            ):
                spin = self._tilt_spins[attr]
                spin.blockSignals(True)
                spin.setValue(variables[var])
                spin.blockSignals(False)
        else:
            self._source_combo.setCurrentIndex(SOURCE_KINDS.index("manual"))
            speed_mph = variables["clubhead_speed_mps"] * MPH_PER_MPS
            updates["clubhead_speed_mph"] = speed_mph
        self._scenario = dataclasses.replace(self._scenario, **updates)
        self._invalidate_source()
        self._tau = None  # auto: impact at maximum clubhead speed
        run = self.run_now()
        offset = variables.get("swing_impact_time_offset_s", 0.0)
        if (
            run is not None
            and run.impact_time_s is not None
            and use_swing_source
            and abs(offset) > 1e-9
        ):
            source = self._ensure_source()
            self._tau = min(max(run.impact_time_s + offset, 0.0), source.duration)
            run = self.run_now()
        return run

    def stop(self) -> None:
        """Stop the playback timer and solver worker (close and tests)."""
        self._view.stop()
        self._solver_panel.stop()

    def _ensure_source(self):  # type: ignore[no-untyped-def]
        if self._source is None:
            config = self.config()
            self._source = make_source(
                config.source_kind,
                self._scenario,
                plane=self.plane(),
                duration=config.swing_duration_s,
                run_config=config.swing_run_config,
                torque_library=config.torque_library,
            )
        return self._source

    def _on_torque_selection_changed(self, *_args: object) -> None:
        """Keep the visible source and cached dynamics aligned with run mode."""
        selection = self._torque_profile_panel.selection()
        if (
            selection.mode is RunMode.PRESCRIBED_TORQUE
            and selection.profile is not None
            and selection.profile.model_id == "model.double_pendulum.v1"
        ):
            self._source_combo.setCurrentIndex(SOURCE_KINDS.index("double_pendulum"))
        self._invalidate_source()

    def _on_joint_locks_changed(self, *_args: object) -> None:
        """Select the compatible kernel whenever an ideal lock is enabled."""
        if self._torque_profile_panel.joint_locks().has_locks:
            self._source_combo.setCurrentIndex(SOURCE_KINDS.index("double_pendulum"))
        self._invalidate_source()

    def _reconcile_joint_locks_for_source(self, *_args: object) -> None:
        """Clear constraints when the user explicitly leaves the supported source."""
        if self.source_kind() != "double_pendulum":
            self._torque_profile_panel.clear_joint_locks(emit=False)

    def _fit_current_run(self, degree: int) -> None:
        """Fit the current retained double-pendulum torque history non-modally."""
        if self._run is None:
            self._torque_profile_panel.set_fit_error(
                "run a double-pendulum simulation first."
            )
            return
        self._torque_profile_panel.fit_current_run(self._run, degree)

    def _on_contact_mode_changed(self, *_args: object) -> None:
        """Reset incompatible impact-time state and explain the active policy."""
        self._tau = None
        self._update_contact_controls()
        self._mark_stale()
        self._emit_config()

    def _update_contact_controls(self) -> None:
        fixed_ball = self.contact_mode() is ContactMode.FIXED_BALL_CONTACT
        if fixed_ball:
            description = (
                "Retains the swing in its original frame and detects sampled "
                "clubhead-reference-point proximity to the fixed ball. A miss is "
                "a valid completed result; mesh contact and swept collision are "
                "not modeled."
            )
        else:
            description = (
                "Forced alignment translates the swing onto the ball at the "
                "selected inspection time. Use this to inspect delivery; it is "
                "not geometric contact detection."
            )
        self._contact_description.setText(description)
        if not hasattr(self, "_scrub_slider"):
            return
        self._scrub_slider.setEnabled(not fixed_ball)
        self._auto_tau_button.setEnabled(not fixed_ball)
        if fixed_ball:
            self._scrub_box.setTitle("Contact Detection (Fixed Ball)")
            self._scrub_label.setText("fixed-ball")
            self._delivery_label.setText(
                "Impact time is detected from sampled fixed-ball proximity; "
                "manual scrubbing is unavailable."
            )
        else:
            self._scrub_box.setTitle("Impact Time (Scrub the Swing Onto the Ball)")
            self._scrub_label.setText("auto")
            self._delivery_label.setText("Awaiting updated simulation")

    def _invalidate_source(self, *_args: object) -> None:
        self._source = None
        # Recompute at maximum speed; tau is source-specific.
        self._tau = None
        # Tilt controls emit before the scrub box exists during construction.
        if hasattr(self, "_scrub_label"):
            self._update_contact_controls()
            self._mark_stale()

    def _scrub_time(self, value: int) -> float:
        source = self._ensure_source()
        return value / _SCRUB_STEPS * float(source.duration)

    def _sync_scrub_slider(self, tau: float) -> None:
        source = self._ensure_source()
        value = (
            round(tau / source.duration * _SCRUB_STEPS) if source.duration > 0.0 else 0
        )
        self._scrub_slider.blockSignals(True)
        self._scrub_slider.setValue(value)
        self._scrub_slider.blockSignals(False)
        self._scrub_label.setText(f"{tau * 1000.0:.1f} ms")

    def _sync_scrub_after_run(self, run: SimulationRun) -> None:
        """Reflect a detected impact or closest-approach sample without fabrication."""
        if run.impact_time_s is not None:
            self._sync_scrub_slider(run.impact_time_s)
            return
        source = self._ensure_source()
        candidate = run.impact_outcome.candidate_time_s
        value = (
            round(candidate / source.duration * _SCRUB_STEPS)
            if source.duration > 0.0
            else 0
        )
        self._scrub_slider.blockSignals(True)
        self._scrub_slider.setValue(value)
        self._scrub_slider.blockSignals(False)
        self._scrub_label.setText(f"closest {candidate * 1000.0:.1f} ms")

    def _update_outcome_labels(self, run: SimulationRun) -> None:
        """Show delivery for hits and proximity diagnostics for misses."""
        if run.impact_time_s is not None:
            self._update_delivery_label(run.impact_time_s)
            return
        outcome = run.impact_outcome
        miss_distance_mm = outcome.closest_approach_m * 1000.0
        threshold_mm = outcome.contact_threshold_m * 1000.0
        self._delivery_label.setText(
            f"No impact detected — closest sampled approach {miss_distance_mm:.1f} "
            f"mm; contact threshold {threshold_mm:.1f} mm."
        )

    def _update_delivery_label(self, tau: float) -> None:
        try:
            source = self._ensure_source()
            delivery = delivery_at(
                source, tau, self._scenario, get_club(self._club_combo.currentText())
            )
        except Exception as exc:  # noqa: BLE001 — zero-speed instants etc.
            self._delivery_label.setText(f"No delivery at this instant ({exc})")
            return
        velocity = delivery.clubhead_velocity
        vx, vy, vz = (float(component) for component in velocity)
        speed_mph = float(np.linalg.norm(velocity)) * MPH_PER_MPS
        path = math.degrees(math.atan2(vz, vx))
        aoa = math.degrees(math.atan2(vy, math.hypot(vx, vz)))
        self._delivery_label.setText(
            f"Delivery at τ: {speed_mph:.1f} mph, path {path:+.1f}°, "
            f"AoA {aoa:+.1f}°, spin loft {delivery.spin_loft_deg:.1f}°"
        )

    def _on_scrub_moved(self, value: int) -> None:
        if self.contact_mode() is ContactMode.FIXED_BALL_CONTACT:
            return
        tau = self._scrub_time(value)
        self._tau = tau
        self._scrub_label.setText(f"{tau * 1000.0:.1f} ms")
        self._update_delivery_label(tau)
        # Dragging updates delivery live; the full impact + flight rerun
        # happens on release (or immediately for programmatic setValue).
        if not self._scrub_slider.isSliderDown() and self._run is not None:
            self.run_now()

    def _on_scrub_released(self) -> None:
        if self.contact_mode() is ContactMode.FIXED_BALL_CONTACT:
            return
        if self._run is not None:
            self.run_now()

    def _on_auto_tau(self) -> None:
        if self.contact_mode() is ContactMode.FIXED_BALL_CONTACT:
            return
        self._tau = None
        self.run_now()

    def _set_completed_status(self, run: SimulationRun) -> None:
        outcome = run.impact_outcome
        lock_summary = self._torque_profile_panel.joint_lock_summary()
        if outcome.is_hit:
            assert run.impact_time_s is not None
            text = (
                f"Completed — Hit at {run.impact_time_s * 1000.0:.1f} ms. "
                "Swing, impact, launch, and flight results are current. "
                f"Joint constraints: {lock_summary}."
            )
            self._set_run_status(text, "hit")
            return
        clearance_mm = -outcome.contact_margin_m * 1000.0
        text = (
            "Completed — No Impact. The closest approach remained "
            f"{clearance_mm:.1f} mm outside the sampled contact threshold. "
            "Swing playback and pendulum kinetics remain available; impact, "
            "launch, and flight values are unavailable. "
            f"Joint constraints: {lock_summary}."
        )
        self._set_run_status(text, "miss")

    def _mark_stale(self) -> None:
        if not hasattr(self, "_run_status"):
            return
        self._set_run_status(
            "Stale — Configuration changed. Run Simulation to refresh results.",
            "stale",
        )

    def _set_run_status(self, text: str, state: str) -> None:
        self._run_status.setText(text)
        self._run_status.setProperty("runState", state)
        self._run_status.setAccessibleDescription(text)

    def _show_explanation(self, field: str) -> None:
        labels = {key: label for key, label, _unit in LAUNCH_ROWS}
        text = LAUNCH_EXPLANATIONS.get(field, "")
        # Persistent single selection across the launch rows (#4120 V4).
        for row_field, row in self._rows.items():
            row.set_selected(row_field == field)
        html = explanation_html(labels.get(field, field), text, field)
        self._explanation.setHtml(html)

    def _on_explanation_link(self, url) -> None:  # type: ignore[no-untyped-def]
        """Forward ``glossary:<term>`` links to the main window."""
        text = url.toString()
        if text.startswith("glossary:"):
            self.glossaryRequested.emit(text.partition(":")[2])
