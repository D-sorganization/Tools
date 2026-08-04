"""The Simulation tab: swing source, plane tilts, club, scrub, run.

Hosts the whole simulation session UI (epic #4103): swing-source picker
(manual scenario / double pendulum / triple pendulum), the three
sequential plane-tilt inputs, a club picker (reusing the shared club
library), the flight-model picker, the Run button, the impact-time
scrubber (the ball is fixed; scrubbing tau translates the swing so the
clubhead at tau meets it, with delivery numbers updating live), the 3D
scene with playback + toggles (:class:`SimulationView`), the launch
result rows with click-through explanations, and the run-data inspector
with CSV/JSON export (:class:`InspectorView`).

Every input carries sourced hover guidance (FIELD_GUIDANCE pattern);
the tab consumes complete scenarios from the main window (LoD) and
builds :class:`SimulationConfig` objects for the session layer.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.club import club_names, get_club
from rate_of_closure.derivation import LAUNCH_EXPLANATIONS
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import (
    SOURCE_KINDS,
    SimulationConfig,
    SimulationRun,
    delivery_at,
    make_source,
    run_simulation,
)
from rate_of_closure.ui.pyqt6.inspector_view import InspectorView
from rate_of_closure.ui.pyqt6.result_row import ResultRow
from rate_of_closure.ui.pyqt6.simulation_view import SimulationView
from rate_of_closure.units import FIELD_GUIDANCE
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.types import PlaneOrientation

logger = logging.getLogger(__name__)

__all__ = ["LAUNCH_ROWS", "SOURCE_LABELS", "SimulationTab"]

#: Source kind -> Title Case combo label (order matches SOURCE_KINDS).
SOURCE_LABELS: dict[str, str] = {
    "manual": "Manual Scenario (Constant Twist)",
    "double_pendulum": "Double Pendulum",
    "triple_pendulum": "Triple Pendulum",
}

#: (launch field, Title Case label, unit suffix) in display order. Every
#: field must have an entry in LAUNCH_EXPLANATIONS (test-enforced).
LAUNCH_ROWS: tuple[tuple[str, str, str], ...] = (
    ("ball_speed_mph", "Ball Speed", " mph"),
    ("launch_angle_deg", "Launch Angle", "°"),
    ("launch_azimuth_deg", "Launch Azimuth", "°"),
    ("spin_rpm", "Total Spin", " rpm"),
    ("carry_m", "Carry Distance", " m"),
    ("max_height_m", "Apex Height", " m"),
    ("flight_time_s", "Flight Time", " s"),
    ("landing_angle_deg", "Landing Angle", "°"),
)

_TILT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("yaw_deg", "Plane Yaw", "plane_yaw_deg"),
    ("side_tilt_deg", "Plane Side Tilt", "plane_side_tilt_deg"),
    ("forward_tilt_deg", "Plane Forward Tilt", "plane_forward_tilt_deg"),
)

_SCRUB_STEPS = 1000


class SimulationTab(QWidget):
    """Simulation session tab (controls left, scene/inspector right)."""

    #: Emitted with the SimulationRun after every successful run.
    runCompleted = pyqtSignal(object)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scenario = ImpactScenario(clubhead_speed_mph=113.0)
        self._run: SimulationRun | None = None
        self._tau: float | None = None  # None = auto (max clubhead speed)
        self._source = None  # cached app-frame source for live scrubbing
        self._rows: dict[str, ResultRow] = {}

        self._view = SimulationView()
        self._inspector = InspectorView()

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_setup_box())
        left_layout.addWidget(self._build_scrub_box())
        left_layout.addWidget(self._build_launch_box())
        left_layout.addWidget(self._build_explanation_box())
        left_layout.addStretch(1)
        left.setMinimumWidth(320)

        right = QTabWidget()
        right.addTab(self._view, "Scene")
        right.addTab(self._inspector, "Inspector")

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._show_explanation(LAUNCH_ROWS[0][0])

    # ── construction ────────────────────────────────────────────────
    def _build_setup_box(self) -> QGroupBox:
        box = QGroupBox("Simulation Setup")
        form = QFormLayout(box)

        self._source_combo = QComboBox()
        self._source_combo.addItems([SOURCE_LABELS[kind] for kind in SOURCE_KINDS])
        self._source_combo.setToolTip(FIELD_GUIDANCE["swing_source"])
        self._source_combo.currentIndexChanged.connect(self._invalidate_source)
        form.addRow("Swing Source", self._source_combo)

        self._tilt_spins: dict[str, QDoubleSpinBox] = {}
        for attr, label, guidance_key in _TILT_SPECS:
            spin = QDoubleSpinBox()
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            spin.setKeyboardTracking(False)
            spin.setDecimals(1)
            spin.setRange(-90.0, 90.0)
            spin.setSuffix(" deg")
            spin.setToolTip(FIELD_GUIDANCE[guidance_key])
            spin.valueChanged.connect(self._invalidate_source)
            self._tilt_spins[attr] = spin
            form.addRow(label, spin)
        self._tilt_spins["side_tilt_deg"].setValue(-45.0)

        self._club_combo = QComboBox()
        self._club_combo.addItems(club_names())
        self._club_combo.setToolTip(FIELD_GUIDANCE["club_selection"])
        form.addRow("Club", self._club_combo)

        self._flight_combo = QComboBox()
        self._flight_combo.addItems([m.value for m in FlightModelType])
        self._flight_combo.setCurrentText("waterloo_penner")
        self._flight_combo.setToolTip(FIELD_GUIDANCE["flight_model"])
        form.addRow("Flight Model", self._flight_combo)

        self._run_button = QPushButton("Run Simulation")
        self._run_button.setToolTip(
            "Generate the swing, solve the impact at the scrubbed instant, "
            "and integrate the ball flight."
        )
        self._run_button.clicked.connect(self.run_now)
        form.addRow(self._run_button)
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
        self._explanation.setMinimumHeight(90)
        self._explanation.setMaximumHeight(150)
        layout.addWidget(self._explanation)
        return box

    # ── public API ──────────────────────────────────────────────────
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

    def config(self) -> SimulationConfig:
        """The simulation request described by the controls."""
        return SimulationConfig(
            scenario=self._scenario,
            club=get_club(self._club_combo.currentText()),
            source_kind=self.source_kind(),
            plane=self.plane(),
            impact_time_s=self._tau,
            flight_model=self._flight_combo.currentText(),
        )

    def run_now(self) -> SimulationRun | None:
        """Run the simulation and populate the scene + inspector."""
        try:
            run = run_simulation(self.config())
        except Exception as exc:  # noqa: BLE001 — surface physics failures
            logger.warning("simulation failed: %s", exc)
            QMessageBox.warning(self, "Simulation Failed", str(exc))
            return None
        self._run = run
        self._tau = run.impact_time_s
        self._sync_scrub_slider(run.impact_time_s)
        self._view.set_run(run)
        self._inspector.set_run(run)
        for field, _label, unit in LAUNCH_ROWS:
            value = run.launch[field]
            text = f"{value:+.1f}{unit}" if math.isfinite(value) else "—"
            self._rows[field].value_label.setText(text)
        self._update_delivery_label(run.impact_time_s)
        self.runCompleted.emit(run)
        return run

    def last_run(self) -> SimulationRun | None:
        """The most recent successful run, if any."""
        return self._run

    def view(self) -> SimulationView:
        """The 3D scene (playback controls live on it)."""
        return self._view

    def inspector(self) -> InspectorView:
        """The run-data inspector."""
        return self._inspector

    def stop(self) -> None:
        """Stop the playback timer (window close and tests)."""
        self._view.stop()

    # ── internals ──────────────────────────────────────────────────
    def _ensure_source(self):  # type: ignore[no-untyped-def]
        if self._source is None:
            self._source = make_source(
                self.source_kind(), self._scenario, plane=self.plane()
            )
        return self._source

    def _invalidate_source(self, *_args: object) -> None:
        self._source = None
        if self._tau is not None:
            self._update_delivery_label(self._tau)

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
        speed_mph = float(np.linalg.norm(velocity)) * 2.2369362920544025
        path = math.degrees(math.atan2(float(velocity[2]), float(velocity[0])))
        aoa = math.degrees(
            math.atan2(
                float(velocity[1]),
                math.hypot(float(velocity[0]), float(velocity[2])),
            )
        )
        self._delivery_label.setText(
            f"Delivery at τ: {speed_mph:.1f} mph, path {path:+.1f}°, "
            f"AoA {aoa:+.1f}°, spin loft {delivery.spin_loft_deg:.1f}°"
        )

    def _on_scrub_moved(self, value: int) -> None:
        tau = self._scrub_time(value)
        self._tau = tau
        self._scrub_label.setText(f"{tau * 1000.0:.1f} ms")
        self._update_delivery_label(tau)
        # Dragging updates delivery live; the full impact + flight rerun
        # happens on release (or immediately for programmatic setValue).
        if not self._scrub_slider.isSliderDown() and self._run is not None:
            self.run_now()

    def _on_scrub_released(self) -> None:
        if self._run is not None:
            self.run_now()

    def _on_auto_tau(self) -> None:
        self._tau = None
        self.run_now()

    def _show_explanation(self, field: str) -> None:
        labels = {key: label for key, label, _unit in LAUNCH_ROWS}
        text = LAUNCH_EXPLANATIONS.get(field, "")
        self._explanation.setHtml(f"<b>{labels.get(field, field)}</b><br/>{text}")
