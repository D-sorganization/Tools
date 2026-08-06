"""Standalone Ball-Flight Explorer tab — launch entry to flight, no swing.

Epic #4120 (V2): type launch conditions directly (ball speed with a
unit drop-down, launch angle, launch direction, spin, spin-axis tilt) OR club
delivery numbers run through ``swing_sim.impact.delivery`` and the
rigid-body impact model, pick any of the 7 literature flight models,
and render the result in the dedicated flight-scale
:class:`~rate_of_closure.ui.pyqt6.flight_view.FlightView` with result
rows (carry, apex, flight time, landing angle, lateral) that click
through to explanations. Every control carries sourced hover guidance.

The physics lives in :mod:`rate_of_closure.simulation.flight_explorer`;
this widget is presentation only.
"""

from __future__ import annotations

import logging
import math

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWhatsThis,
    QWidget,
)

from rate_of_closure.derivation import LAUNCH_EXPLANATIONS
from rate_of_closure.model import MPH_PER_MPS
from rate_of_closure.simulation import (
    FlightExploration,
    WindComparison,
    explore_with_optional_wind,
    launch_from_delivery,
    launch_from_direct,
)
from rate_of_closure.ui.pyqt6.flight_view import FlightView
from rate_of_closure.ui.pyqt6.flight_wind_controls import FlightWindControls
from rate_of_closure.ui.pyqt6.result_row import ResultRow, explanation_html
from rate_of_closure.units import FIELD_GUIDANCE, format_distance_m
from shared.python.swing_sim.flight import (
    LAUNCH_DIRECTION_DEFINITIONS,
    LaunchDirectionConvention,
    launch_direction_sign_labels,
)
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.impact import DeliveryParameters

logger = logging.getLogger(__name__)

__all__ = ["EXPLORER_ROWS", "FlightExplorerTab"]

#: Entry modes, in combo order.
_MODES: tuple[str, ...] = ("Direct Launch Conditions", "Impact Delivery")

#: Speed display units: label -> factor from displayed to m/s.
_SPEED_UNITS: dict[str, float] = {"mph": 1.0 / MPH_PER_MPS, "m/s": 1.0}

#: (metric key, Title Case label, unit suffix) result rows in display
#: order. Every key must have an entry in LAUNCH_EXPLANATIONS
#: (test-enforced).
EXPLORER_ROWS: tuple[tuple[str, str, str], ...] = (
    ("carry_m", "Carry Distance", " m"),
    ("max_height_m", "Apex Height", " m"),
    ("flight_time_s", "Flight Time", " s"),
    ("landing_angle_deg", "Landing Angle", "°"),
    ("lateral_m", "Lateral Landing Offset", " m"),
)

#: Rows following the user's distance display unit (#4125 H6). Apex
#: stays in the metres height convention deliberately.
_DISTANCE_ROWS: frozenset[str] = frozenset({"carry_m", "lateral_m"})

#: Direct-mode fields: (attr, label, guidance key, low, high, default,
#: decimals, suffix).
_DIRECT_FIELDS: tuple[tuple[str, str, str, float, float, float, int, str], ...] = (
    ("launch_angle_deg", "Launch Angle", "fx_launch_angle", -89.0, 89.0, 10.9, 1, "°"),
    (
        "launch_direction_deg",
        "Launch Direction",
        "fx_launch_direction",
        -45.0,
        45.0,
        0.0,
        1,
        "°",
    ),
    ("spin_rpm", "Total Spin", "fx_spin_rpm", 0.0, 15000.0, 2686.0, 0, " rpm"),
    (
        "spin_axis_tilt_deg",
        "Spin-Axis Tilt",
        "fx_spin_axis_tilt",
        -60.0,
        60.0,
        0.0,
        1,
        "°",
    ),
)

#: Delivery-mode fields: same tuple shape as _DIRECT_FIELDS.
_DELIVERY_FIELDS: tuple[tuple[str, str, str, float, float, float, int, str], ...] = (
    ("club_path_deg", "Club Path", "fx_club_path", -45.0, 45.0, 0.0, 1, "°"),
    ("face_angle_deg", "Face Angle", "fx_face_angle", -45.0, 45.0, 0.0, 1, "°"),
    ("attack_angle_deg", "Attack Angle", "fx_attack_angle", -20.0, 20.0, -1.0, 1, "°"),
    ("dynamic_loft_deg", "Dynamic Loft", "fx_dynamic_loft", 0.0, 70.0, 12.0, 1, "°"),
    (
        "impact_offset_toe_mm",
        "Impact Toward Toe",
        "impact_offset_toe_mm",
        -30.0,
        30.0,
        0.0,
        1,
        " mm",
    ),
    (
        "impact_offset_high_mm",
        "Impact Above Center",
        "impact_offset_high_mm",
        -30.0,
        30.0,
        0.0,
        1,
        " mm",
    ),
)


def _make_spin(
    low: float, high: float, default: float, decimals: int, suffix: str, tooltip: str
) -> QDoubleSpinBox:
    """A typed entry spin box in the explorer's house style."""
    spin = QDoubleSpinBox()
    spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spin.setKeyboardTracking(False)
    spin.setDecimals(decimals)
    spin.setRange(low, high)
    spin.setValue(default)
    spin.setSuffix(suffix)
    spin.setToolTip(tooltip)
    spin.setMinimumWidth(84)  # stays readable at small windows
    return spin


def _field_label(label: str, attr: str, guidance: str) -> QWidget:
    """Return a visibly clickable field label with non-modal guidance."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(QLabel(label))
    button = QToolButton()
    button.setText("Details")
    button.setAutoRaise(True)
    button.setObjectName(f"{attr.removesuffix('_deg')}_info")
    button.setAccessibleName(f"Explain {label}")
    button.setAccessibleDescription(guidance)
    button.setToolTip(guidance)
    button.clicked.connect(
        lambda _checked=False: QWhatsThis.showText(
            button.mapToGlobal(button.rect().bottomLeft()), guidance, button
        )
    )
    row.addWidget(button)
    return container


class FlightExplorerTab(QWidget):
    """Standalone flight explorer: launch entry, model picker, viewer."""

    #: Emitted with a glossary term key when an explanation link is used.
    glossaryRequested = pyqtSignal(str)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._exploration: FlightExploration | None = None
        self.wind_comparison: WindComparison | None = None
        self._rows: dict[str, ResultRow] = {}
        self._direct_spins: dict[str, QDoubleSpinBox] = {}
        self._delivery_spins: dict[str, QDoubleSpinBox] = {}
        self._flight_view = FlightView()

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(self._build_entry_box())
        self.wind_controls = FlightWindControls()
        left_layout.addWidget(self.wind_controls)
        left_layout.addWidget(self._build_results_box())
        left_layout.addWidget(self._build_explanation_box())
        left_layout.addStretch(1)
        left = QScrollArea()
        left.setWidgetResizable(True)
        left.setFrameShape(QFrame.Shape.NoFrame)
        left.setWidget(left_content)
        left.setMinimumWidth(300)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._flight_view)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._show_explanation(EXPLORER_ROWS[0][0])

    # ── construction ────────────────────────────────────────────────
    def _build_entry_box(self) -> QGroupBox:
        box = QGroupBox("Launch Entry (No Swing Required)")
        layout = QVBoxLayout(box)
        form = QFormLayout()

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(list(_MODES))
        self._mode_combo.setToolTip(FIELD_GUIDANCE["fx_mode"])
        form.addRow("Entry Mode", self._mode_combo)

        speed_row = QHBoxLayout()
        self._speed_spin = _make_spin(
            1.0, 250.0, 167.0, 1, "", FIELD_GUIDANCE["fx_ball_speed"]
        )
        self._speed_unit_combo = QComboBox()
        self._speed_unit_combo.addItems(list(_SPEED_UNITS))
        self._speed_unit_combo.setToolTip(FIELD_GUIDANCE["fx_speed_unit"])
        self._speed_unit_combo.currentTextChanged.connect(self._on_speed_unit)
        self._speed_unit = "mph"
        speed_row.addWidget(self._speed_spin, stretch=1)
        speed_row.addWidget(self._speed_unit_combo)
        form.addRow("Speed", speed_row)
        layout.addLayout(form)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_fields_page(_DIRECT_FIELDS, "direct"))
        self._stack.addWidget(self._build_fields_page(_DELIVERY_FIELDS, "delivery"))
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        layout.addWidget(self._stack)

        direction_form = QFormLayout()
        self._direction_convention_combo = QComboBox()
        self._direction_convention_combo.addItem(
            "App Native (+ Right)", LaunchDirectionConvention.APP_NATIVE
        )
        self._direction_convention_combo.addItem(
            "TrackMan-Comparable (+ Right)",
            LaunchDirectionConvention.TRACKMAN_COMPARABLE,
        )
        self._direction_convention_combo.setAccessibleName(
            "Launch Direction Convention"
        )
        self._direction_convention_combo.setToolTip(
            "Choose how entered Launch Direction values are interpreted."
        )
        self._direction_convention_combo.currentIndexChanged.connect(
            self._refresh_direction_example
        )
        direction_form.addRow("Direction Convention", self._direction_convention_combo)
        self._direction_example = QLabel()
        self._direction_example.setWordWrap(True)
        self._direction_example.setAccessibleName("Launch Direction Sign Example")
        direction_form.addRow("", self._direction_example)
        layout.addLayout(direction_form)
        self._refresh_direction_example()

        model_form = QFormLayout()
        self._model_combo = QComboBox()
        self._model_combo.addItems([m.value for m in FlightModelType])
        self._model_combo.setCurrentText("waterloo_penner")
        self._model_combo.setToolTip(FIELD_GUIDANCE["flight_model"])
        model_form.addRow("Flight Model", self._model_combo)
        layout.addLayout(model_form)

        self._run_button = QPushButton("Run Flight")
        self._run_button.setToolTip(
            "Build launch conditions from the entries (running the "
            "impact model first in Impact Delivery mode) and integrate "
            "the ball flight with the selected literature model."
        )
        self._run_button.clicked.connect(self.run_now)
        layout.addWidget(self._run_button)
        return box

    def _build_fields_page(
        self,
        fields: tuple[tuple[str, str, str, float, float, float, int, str], ...],
        kind: str,
    ) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        target = self._direct_spins if kind == "direct" else self._delivery_spins
        for attr, label, guidance_key, low, high, default, decimals, suffix in fields:
            spin = _make_spin(
                low, high, default, decimals, suffix, FIELD_GUIDANCE[guidance_key]
            )
            target[attr] = spin
            form.addRow(_field_label(label, attr, FIELD_GUIDANCE[guidance_key]), spin)
        return page

    def _build_results_box(self) -> QGroupBox:
        box = QGroupBox("Flight Numbers")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        for key, label, _unit in EXPLORER_ROWS:
            row = ResultRow(key, label)
            row.clicked.connect(self._show_explanation)
            self._rows[key] = row
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

    # ── public API ──────────────────────────────────────────────────
    def speed_mps(self) -> float:
        """The entered speed converted to m/s."""
        return self._speed_spin.value() * _SPEED_UNITS[self._speed_unit]

    def mode(self) -> str:
        """The selected entry mode label."""
        return _MODES[self._mode_combo.currentIndex()]

    def flight_view(self) -> FlightView:
        """The embedded flight-scale viewer."""
        return self._flight_view

    def last_exploration(self) -> FlightExploration | None:
        """The most recent successful exploration, if any."""
        return self._exploration

    def run_now(self) -> FlightExploration | None:
        """Build launch conditions, run the flight, populate the views."""
        try:
            if self.mode() == _MODES[0]:
                launch = launch_from_direct(
                    ball_speed_mph=self.speed_mps() * MPH_PER_MPS,
                    launch_angle_deg=self._direct_spins["launch_angle_deg"].value(),
                    launch_direction_deg=self._direct_spins[
                        "launch_direction_deg"
                    ].value(),
                    spin_rpm=self._direct_spins["spin_rpm"].value(),
                    spin_axis_tilt_deg=self._direct_spins["spin_axis_tilt_deg"].value(),
                    direction_convention=self._direction_convention_combo.currentData(),
                )
            else:
                launch = launch_from_delivery(
                    DeliveryParameters(
                        clubhead_speed_mps=self.speed_mps(),
                        club_path_deg=self._delivery_spins["club_path_deg"].value(),
                        face_angle_deg=self._delivery_spins["face_angle_deg"].value(),
                        attack_angle_deg=self._delivery_spins[
                            "attack_angle_deg"
                        ].value(),
                        dynamic_loft_deg=self._delivery_spins[
                            "dynamic_loft_deg"
                        ].value(),
                        impact_offset_toe_mm=self._delivery_spins[
                            "impact_offset_toe_mm"
                        ].value(),
                        impact_offset_high_mm=self._delivery_spins[
                            "impact_offset_high_mm"
                        ].value(),
                    )
                )
            model_name = self._model_combo.currentText()
            exploration, comparison = explore_with_optional_wind(
                launch, self.wind_controls.optional_scenario(), model_name
            )
        except Exception as exc:  # noqa: BLE001 — surface physics failures
            logger.warning("flight exploration failed: %s", exc)
            QMessageBox.warning(self, "Flight Failed", str(exc))
            return None
        self._exploration = exploration
        self.wind_comparison = comparison
        self.wind_controls.set_comparison(comparison)
        self._flight_view.set_trajectory(exploration.positions)
        calm = None if comparison is None else comparison.calm.positions
        self._flight_view.set_comparison_trajectory(calm)
        self._refresh_rows()
        return exploration

    def _refresh_rows(self) -> None:
        """Format the result rows; carry/lateral follow the distance
        display unit (#4125 H6 — yards default, apex stays metres)."""
        if self._exploration is None:
            return
        for key, _label, unit in EXPLORER_ROWS:
            value = self._exploration.metrics[key]
            if not math.isfinite(value):
                text = "—"
            elif key in _DISTANCE_ROWS:
                text = (
                    f"+{format_distance_m(value)}"
                    if value >= 0
                    else (f"-{format_distance_m(-value)}")
                )
            else:
                text = f"{value:+.1f}{unit}"
            self._rows[key].value_label.setText(text)

    def refresh_units(self) -> None:
        """Re-render distance rows after a display-unit change."""
        self._refresh_rows()

    # ── internals ──────────────────────────────────────────────────
    def _on_mode_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        label = "Ball Speed" if index == 0 else "Clubhead Speed"
        tooltip = (
            FIELD_GUIDANCE["fx_ball_speed"]
            if index == 0
            else FIELD_GUIDANCE["clubhead_speed_mph"]
        )
        self._speed_spin.setToolTip(tooltip)
        self._run_button.setText("Run Flight" if index == 0 else "Run Impact + Flight")
        logger.debug("flight-explorer mode -> %s (%s)", self.mode(), label)

    def _on_speed_unit(self, unit: str) -> None:
        previous = self._speed_unit
        if unit == previous:
            return
        mps = self._speed_spin.value() * _SPEED_UNITS[previous]
        self._speed_unit = unit
        self._speed_spin.blockSignals(True)
        self._speed_spin.setValue(mps / _SPEED_UNITS[unit])
        self._speed_spin.blockSignals(False)

    def _refresh_direction_example(self) -> None:
        convention = self._direction_convention_combo.currentData()
        definition = LAUNCH_DIRECTION_DEFINITIONS[convention]
        positive, negative = launch_direction_sign_labels(convention)
        self._direction_example.setText(
            f"0° = straight · + = {positive} · − = {negative} · "
            f"{definition.quantity_status.value}"
        )

    def _show_explanation(self, key: str) -> None:
        labels = {row_key: label for row_key, label, _unit in EXPLORER_ROWS}
        text = LAUNCH_EXPLANATIONS.get(key, "")
        # Persistent single selection across the result rows (#4120 V4).
        for row_field, row in self._rows.items():
            row.set_selected(row_field == key)
        self._explanation.setHtml(explanation_html(labels.get(key, key), text, key))

    def _on_explanation_link(self, url) -> None:  # type: ignore[no-untyped-def]
        """Forward ``glossary:<term>`` links to the main window."""
        text = url.toString()
        if text.startswith("glossary:"):
            self.glossaryRequested.emit(text.partition(":")[2])
