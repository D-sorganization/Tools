"""Simulation session UI for source setup, playback, results, and inspection."""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import cast

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.club import ClubSpec, get_club
from rate_of_closure.derivation_models import DerivationConfig
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    SOURCE_KINDS,
    BallSetup,
    ContactMode,
    SimulationConfig,
    SimulationRun,
    run_simulation,
)
from rate_of_closure.ui.pyqt6.ball_setup_control import BallSetupControl
from rate_of_closure.ui.pyqt6.flight_playback_controls import FlightPlaybackPanel
from rate_of_closure.ui.pyqt6.flight_view import FlightView
from rate_of_closure.ui.pyqt6.inspector_view import InspectorView
from rate_of_closure.ui.pyqt6.kinetics_panel import KineticsPanel
from rate_of_closure.ui.pyqt6.result_row import ResultRow
from rate_of_closure.ui.pyqt6.simulation_specs import (
    LAUNCH_ROWS,
    SOURCE_LABELS,
)
from rate_of_closure.ui.pyqt6.simulation_tab_compositor import (
    SimulationTabCompositorMixin,
)
from rate_of_closure.ui.pyqt6.simulation_tab_controls import (
    SimulationTabControlsMixin,
)
from rate_of_closure.ui.pyqt6.simulation_tab_runtime import SimulationTabRuntimeMixin
from rate_of_closure.ui.pyqt6.simulation_target_workflow import (
    SimulationTargetWorkflowMixin,
)
from rate_of_closure.ui.pyqt6.simulation_view import SimulationView
from rate_of_closure.ui.pyqt6.simulation_workspace_bridge import (
    SimulationWorkspaceBridgeMixin,
)
from rate_of_closure.ui.pyqt6.solver_panel import SolverPanel
from rate_of_closure.ui.pyqt6.strike_view import StrikeView
from rate_of_closure.ui.pyqt6.synchronized_simulation_view import (
    SynchronizedSimulationView,
)
from rate_of_closure.ui.pyqt6.torque_profile_controller import RunMode
from rate_of_closure.ui.pyqt6.torque_profile_panel import TorqueProfilePanel
from rate_of_closure.ui.pyqt6.view_compositor import ViewCompositor
from rate_of_closure.units import format_distance_m
from rate_of_closure.view_workspace import ViewKind
from shared.python.swing_sim.run_config import DoublePendulumRunConfig
from shared.python.swing_sim.types import PlaneOrientation

logger = logging.getLogger(__name__)

__all__ = ["LAUNCH_ROWS", "SOURCE_LABELS", "SimulationTab"]


class SimulationTab(
    SimulationTabControlsMixin,
    SimulationTabCompositorMixin,
    SimulationTabRuntimeMixin,
    SimulationTargetWorkflowMixin,
    SimulationWorkspaceBridgeMixin,
    QWidget,
):
    """Simulation session tab (controls left, scene/inspector right)."""

    #: Emitted with the SimulationRun after every successful run.
    runCompleted = pyqtSignal(object)  # noqa: N815 - Qt signal convention
    #: Emitted with a glossary term key when an explanation link is used.
    glossaryRequested = pyqtSignal(str)  # noqa: N815 - Qt signal convention
    #: Drives conditional Calculation Description sections from model changes.
    configChanged = pyqtSignal(object)  # noqa: N815 - Qt signal convention
    #: Requests that the owning workbench adopt a library-club selection.
    clubSelectionChanged = pyqtSignal(str)  # noqa: N815 - Qt signal convention

    def __init__(
        self,
        parent: QWidget | None = None,
        view_settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._scenario = ImpactScenario(clubhead_speed_mph=113.0)
        self._run: SimulationRun | None = None
        self._run_current = False
        self._tau: float | None = None  # None = auto (max clubhead speed)
        self._source = None  # cached app-frame source for live scrubbing
        self._rows: dict[str, ResultRow] = {}

        self._view = SimulationView()
        self._strike_view = StrikeView()
        self._flight_view = FlightView()
        self._flight_panel = FlightPlaybackPanel(self._flight_view)
        self._compositor_strike_view = StrikeView()
        self._compositor_swing_view = SynchronizedSimulationView()
        self._compositor_flight_view = FlightView()
        self._compositor = ViewCompositor(
            {
                ViewKind.IMPACT: self._compositor_strike_view,
                ViewKind.SWING: self._compositor_swing_view,
                ViewKind.FLIGHT: self._compositor_flight_view,
            },
            view_settings,
        )
        self._compositor_swing_view.playbackTimeChanged.connect(
            self._sync_compositor_playback
        )
        restored_playback = self._compositor.workspace().playback
        self._compositor_swing_view.set_looping(restored_playback.loop)
        self._compositor_swing_view.set_playback_rate(restored_playback.rate)
        self._kinetics_panel = KineticsPanel()
        self._kinetics_panel.glossaryRequested.connect(self.glossaryRequested)
        self._inspector = InspectorView()
        self._inspector.settingsImported.connect(self._on_imported_simulation_settings)
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
        left_content.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(self._build_setup_box())
        left_layout.addWidget(self._build_spatial_target_control())
        self._scrub_box = self._build_scrub_box()
        left_layout.addWidget(self._scrub_box)
        left_layout.addWidget(self._build_launch_box())
        left_layout.addWidget(self._build_explanation_box())
        left_layout.addStretch(1)
        # Scrolling preserves readable entries in small windows.
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.Shape.NoFrame)
        left.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left.setWidget(left_content)
        left.setMinimumWidth(310)
        self._controls_scroll = left

        # Scale-separated face, swing, kinetics, and flight displays.
        right = QTabWidget()
        right.addTab(self._strike_view, "Strike")
        right.addTab(self._view, "Swing")
        right.addTab(self._kinetics_panel, "Kinetics")
        right.addTab(self._flight_panel, "Flight")
        right.addTab(self._compositor, "Multi View")
        right.addTab(self._inspector, "Inspector")
        right.addTab(self._solver_panel, "Solver")
        right.addTab(self._torque_profile_panel, "Torque Profiles")
        right.setCurrentWidget(self._view)
        right.setElideMode(Qt.TextElideMode.ElideRight)
        right.setDocumentMode(True)
        self._display_tabs = right

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)
        splitter.setSizes([320, 720])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._show_explanation(LAUNCH_ROWS[0][0])

    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Adopt the explorer's scenario (drives the manual source)."""
        self._scenario = scenario
        self._invalidate_source()
        self.configChanged.emit(self.derivation_config())

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
        """Adopt one library club and keep its geometry internally coherent."""
        club = get_club(name)
        self._club_spec = club
        self._scenario = dataclasses.replace(
            self._scenario,
            lie_angle_deg=club.lie_deg,
            com_to_face_mm=club.cg_depth_m * 1000.0,
        )
        default_setup = SimulationConfig(scenario=self._scenario, club=club).ball_setup
        self._ball_setup_control.apply_club_default(default_setup, club.name)
        self.clubSelectionChanged.emit(name)
        self._emit_config()

    def set_club_spec(self, club: ClubSpec) -> None:
        """Apply the owning workbench's complete club spec without signal loops."""
        if not isinstance(club, ClubSpec):
            raise TypeError("club must be a ClubSpec")
        self._club_spec = club
        self._club_combo.blockSignals(True)
        self._club_combo.setCurrentText(club.name)
        self._club_combo.blockSignals(False)
        default_setup = SimulationConfig(scenario=self._scenario, club=club).ball_setup
        self._ball_setup_control.apply_club_default(default_setup, club.name)
        self._emit_config()

    def contact_mode(self) -> ContactMode:
        """The selected contact policy."""
        return cast(ContactMode, self._contact_combo.currentData())

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
            club=self._club_spec,
            ball_setup=self._ball_setup_control.setup(),
            source_kind=source_kind,
            plane=self.plane(),
            manual_attack_angle_deg=self._manual_delivery_spins[
                "attack_angle_deg"
            ].value(),
            manual_club_path_deg=self._manual_delivery_spins["club_path_deg"].value(),
            manual_forward_shaft_lean_deg=self._manual_delivery_spins[
                "forward_shaft_lean_deg"
            ].value(),
            manual_shaft_axis_datum=self._shaft_datum_combo.currentData(),
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
        self._run_current = False
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
        self._compositor_swing_view.set_run(run)
        self._compositor_strike_view.set_run(run)
        self._compositor_flight_view.set_run(run)
        self._sync_compositor_playback(self._compositor_swing_view.playback_time())
        self._inspector.set_run(run)
        self._refresh_launch_rows()
        self._update_spatial_target_after_run(run)
        self._update_outcome_labels(run)
        self._set_completed_status(run)
        self._run_current = True
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

    def current_completed_hit(self) -> SimulationRun | None:
        """Return only a successful hit that still matches every current editor."""
        run = self._run
        if not self._run_current or run is None or not run.impact_outcome.is_hit:
            return None
        return run

    def ball_setup_control(self) -> BallSetupControl:
        """Return the canonical Ground/Tee editor hosted by this session."""
        return self._ball_setup_control

    def set_ball_setup(self, setup: BallSetup) -> None:
        """Load a canonical persisted setup without introducing a UI schema."""
        self._ball_setup_control.set_setup(setup)
        self._emit_config()

    def kinetics_panel(self) -> KineticsPanel:
        """The kinetics plots + peak-table sub-tab (#4125 H2)."""
        return self._kinetics_panel

    def inspector(self) -> InspectorView:
        """The run-data inspector."""
        return self._inspector

    def solver_panel(self) -> SolverPanel:
        """The goal-driven Solver panel (worker lifecycle lives on it)."""
        return self._solver_panel

    def stop(self) -> None:
        """Stop the playback timer and solver worker (close and tests)."""
        self._view.stop()
        self._compositor_swing_view.stop()
        self._solver_panel.stop()
