"""Construction helpers for the PyQt simulation-session controls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
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
    QSlider,
    QTextBrowser,
    QVBoxLayout,
)

from rate_of_closure.club import club_names, get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import SOURCE_KINDS, ContactMode, SimulationConfig
from rate_of_closure.ui.pyqt6.ball_setup_control import BallSetupControl
from rate_of_closure.ui.pyqt6.result_row import ResultRow
from rate_of_closure.ui.pyqt6.simulation_specs import (
    LAUNCH_ROWS,
    SOURCE_LABELS,
    TILT_SPECS,
)
from rate_of_closure.units import FIELD_GUIDANCE
from shared.python.swing_sim.flight.registry import FlightModelType

SCRUB_STEPS = 1000


class SimulationTabControlsMixin:
    """Build controls while leaving simulation orchestration to the host tab."""

    _scenario: ImpactScenario
    _rows: dict[str, ResultRow]
    _scrub_box: QGroupBox

    if TYPE_CHECKING:

        def _emit_config(self, *_args: object) -> None: ...

        def _invalidate_source(self, *_args: object) -> None: ...

        def _on_auto_tau(self) -> None: ...

        def _on_club_changed(self, name: str) -> None: ...

        def _on_contact_mode_changed(self, *_args: object) -> None: ...

        def _on_explanation_link(self, url: object) -> None: ...

        def _on_scrub_moved(self, value: int) -> None: ...

        def _on_scrub_released(self) -> None: ...

        def _reconcile_joint_locks_for_source(self, *_args: object) -> None: ...

        def _show_explanation(self, field: str) -> None: ...

        def _update_contact_controls(self) -> None: ...

        def run_now(self) -> object: ...

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
            spin.setMinimumWidth(84)
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
        self._flight_combo.addItems([model.value for model in FlightModelType])
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
        self._scrub_slider.setRange(0, SCRUB_STEPS)
        self._scrub_slider.setValue(SCRUB_STEPS // 2)
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
