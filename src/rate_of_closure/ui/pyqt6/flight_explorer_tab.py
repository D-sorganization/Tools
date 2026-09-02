"""Standalone launch-to-flight explorer with plots and 3D playback.

The presentation widget combines direct/delivery launch entry, seven literature
flight models, wind-pair comparison, canonical spatial targets, result
explanations, and timestamp-accurate 3D playback. Physics remains in
``rate_of_closure.simulation.flight_explorer``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.derivation import LAUNCH_EXPLANATIONS
from rate_of_closure.flight_accepted_study import AcceptedFlightStudy
from rate_of_closure.model import MPH_PER_MPS
from rate_of_closure.simulation import FlightExploration, WindComparison
from rate_of_closure.simulation.flight_record_playback import (
    timed_trajectory_from_ball_flight_record,
)
from rate_of_closure.ui.pyqt6.flight_explorer_controls import (
    DELIVERY_FIELDS,
    DIRECT_FIELDS,
    ENTRY_MODES,
    EXPLORER_ROWS,
    field_label,
    make_spin,
)
from rate_of_closure.ui.pyqt6.flight_explorer_run import FlightExplorerRunMixin
from rate_of_closure.ui.pyqt6.flight_playback_controls import FlightPlaybackPanel
from rate_of_closure.ui.pyqt6.flight_view import FlightView
from rate_of_closure.ui.pyqt6.flight_wind_controls import FlightWindControls
from rate_of_closure.ui.pyqt6.result_row import ResultRow, explanation_html
from rate_of_closure.ui.pyqt6.spatial_target_workflow import (
    build_spatial_target_workflow,
)
from rate_of_closure.units import FIELD_GUIDANCE, SPEED_UNITS
from shared.python.swing_sim.flight import (
    LAUNCH_DIRECTION_DEFINITIONS,
    LaunchDirectionConvention,
    launch_direction_sign_labels,
)
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.flight_interchange import ball_flight_trajectory_from_json

logger = logging.getLogger(__name__)

__all__ = ["EXPLORER_ROWS", "FlightExplorerTab"]


class FlightExplorerTab(FlightExplorerRunMixin, QWidget):
    """Standalone flight explorer: launch entry, model picker, viewer."""

    #: Emitted with a glossary term key when an explanation link is used.
    glossaryRequested = pyqtSignal(str)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._exploration: FlightExploration | None = None
        self.wind_comparison: WindComparison | None = None
        self._accepted: AcceptedFlightStudy | None = None
        self._generation = 0
        self._error_origin: str | None = None
        self._rows: dict[str, ResultRow] = {}
        self._direct_spins: dict[str, QDoubleSpinBox] = {}
        self._delivery_spins: dict[str, QDoubleSpinBox] = {}
        self._flight_view = FlightView()
        self._flight_panel = FlightPlaybackPanel(self._flight_view)
        self._flight_view.sampleSelected.connect(self._on_sample_selected)
        self._flight_view.sampleSelectionFailed.connect(self._show_sample_error)

        left_content = QWidget()
        left_layout = QVBoxLayout(left_content)
        left_layout.addWidget(self._build_entry_box())
        self.wind_controls = FlightWindControls()
        left_layout.addWidget(self.wind_controls)
        self._spatial_target_panel, self._target_workflow = (
            build_spatial_target_workflow(self._flight_view)
        )
        left_layout.addWidget(self._spatial_target_panel)
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
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._error_status = QLabel()
        self._error_status.setAccessibleName("Flight explorer error")
        self._error_status.setWordWrap(True)
        self._error_status.setStyleSheet("color: #ef4444")
        self._error_status.setFixedHeight(64)
        self._context_status = QLabel("No accepted flight is available.")
        self._context_status.setAccessibleName("Displayed flight context")
        self._context_status.setWordWrap(True)
        self._sample_status = QLabel(
            "Select the current primary trajectory; calm ghost is comparison-only."
        )
        self._sample_status.setAccessibleName("Selected flight sample")
        self._sample_status.setWordWrap(True)
        self._import_button = QPushButton("Import Trajectory Record…")
        self._import_button.setAccessibleName("Import Trajectory Record")
        self._import_button.setToolTip(
            "Load a swing_sim.ball_flight_trajectory/1 record (ADR-0047) from "
            "either flight-model family and replay it on this 3D playback "
            "through the existing transport."
        )
        self._import_button.clicked.connect(self._import_trajectory_record)
        right_layout.addWidget(self._error_status)
        right_layout.addWidget(self._context_status)
        right_layout.addWidget(self._sample_status)
        right_layout.addWidget(self._import_button)
        right_layout.addWidget(self._flight_panel, stretch=1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._show_explanation(EXPLORER_ROWS[0][0])
        self._connect_identity_editors()

    # ── construction ────────────────────────────────────────────────
    def _build_entry_box(self) -> QGroupBox:
        box = QGroupBox("Launch Entry (No Swing Required)")
        layout = QVBoxLayout(box)
        form = QFormLayout()

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(list(ENTRY_MODES))
        self._mode_combo.setToolTip(FIELD_GUIDANCE["fx_mode"])
        form.addRow("Entry Mode", self._mode_combo)

        speed_row = QHBoxLayout()
        self._speed_spin = make_spin(
            1.0, 250.0, 167.0, 1, "", FIELD_GUIDANCE["fx_ball_speed"]
        )
        self._speed_spin.setAccessibleName("Ball Speed")
        self._speed_unit_combo = QComboBox()
        self._speed_unit_combo.setAccessibleName("Ball Speed Unit")
        self._speed_unit_combo.addItems(list(SPEED_UNITS))
        self._speed_unit_combo.setToolTip(FIELD_GUIDANCE["fx_speed_unit"])
        self._speed_unit_combo.currentTextChanged.connect(self._on_speed_unit)
        self._speed_unit = "mph"
        self._speed_mph = 167.0
        self._speed_spin.valueChanged.connect(self._on_speed_value_changed)
        speed_row.addWidget(self._speed_spin, stretch=1)
        speed_row.addWidget(self._speed_unit_combo)
        form.addRow("Speed", speed_row)
        layout.addLayout(form)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_fields_page(DIRECT_FIELDS, "direct"))
        self._stack.addWidget(self._build_fields_page(DELIVERY_FIELDS, "delivery"))
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
        self._direction_convention_combo.addItem(
            "Foresight-Comparable (Sign Unavailable)"
        )
        foresight_item = cast(
            QStandardItemModel, self._direction_convention_combo.model()
        ).item(2)
        assert foresight_item is not None
        foresight_item.setEnabled(False)
        foresight_item.setToolTip(
            "Unavailable: the general public sign convention is not established "
            "independently of player handedness."
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
            spin = make_spin(
                low, high, default, decimals, suffix, FIELD_GUIDANCE[guidance_key]
            )
            spin.setAccessibleName(label)
            target[attr] = spin
            form.addRow(field_label(label, attr, FIELD_GUIDANCE[guidance_key]), spin)
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
        return float(self._speed_mph / MPH_PER_MPS)

    def speed_mph(self) -> float:
        """The exact canonical speed authority in miles per hour."""
        return self._speed_mph

    def mode(self) -> str:
        """The selected entry mode label."""
        return str(ENTRY_MODES[self._mode_combo.currentIndex()])

    def flight_view(self) -> FlightView:
        """The embedded flight-scale viewer."""
        return self._flight_view

    def last_exploration(self) -> FlightExploration | None:
        """The most recent successful exploration, if any."""
        return None if self._accepted is None else self._accepted.exploration

    def accepted_study(self) -> AcceptedFlightStudy | None:
        """The complete immutable accepted authority, if one has committed."""
        return self._accepted

    # ── internals ──────────────────────────────────────────────────
    def _import_trajectory_record(self) -> None:
        """Import a ``ball_flight_trajectory/1`` record (ADR-0047 H4).

        Wiring only: the record is parsed and frame-converted by
        :mod:`~rate_of_closure.simulation.flight_record_playback`, and
        the resulting samples are handed to
        :meth:`FlightView.set_timed_trajectory`, the same entry point
        every solver-produced flight already uses — no new transport,
        scrub, or speed logic is introduced here.
        """
        path, _filter = QFileDialog.getOpenFileName(
            self, "Import Trajectory Record", "", "Ball Flight Trajectory (*.json)"
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
            record = ball_flight_trajectory_from_json(text)
            timed = timed_trajectory_from_ball_flight_record(record)
        except Exception as exc:
            self._show_error(exc, origin="import")
            return
        self._accepted = None
        self._exploration = None
        self.wind_comparison = None
        self._generation = 0
        for row in self._rows.values():
            row.value_label.setText("—")
        self.wind_controls.set_comparison(None)
        self._flight_view.set_timed_trajectory(timed.times_s, timed.positions_m)
        self._flight_panel.controls.jump_to_time(0.0)
        self._context_status.setText(
            f"Imported trajectory: {record.provenance.model_family} / "
            f"{record.provenance.model_name} (source {record.source_id})"
        )
        self._sample_status.setText(
            "Imported trajectory record — sample inspection is unavailable."
        )
        self._error_status.clear()
        self._error_origin = None

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
        self._speed_unit = unit
        self._speed_spin.blockSignals(True)
        mph_per_display_unit = SPEED_UNITS[unit]
        self._speed_spin.setDecimals(9)
        self._speed_spin.setRange(
            1.0 / mph_per_display_unit,
            250.0 / mph_per_display_unit,
        )
        self._speed_spin.setValue(self._speed_mph / mph_per_display_unit)
        self._speed_spin.blockSignals(False)

    def _on_speed_value_changed(self, value: float) -> None:
        self._speed_mph = value * SPEED_UNITS[self._speed_unit]

    def _connect_identity_editors(self) -> None:
        self._speed_spin.valueChanged.connect(self._mark_inputs_changed)
        for spin in self._direct_spins.values():
            spin.valueChanged.connect(self._mark_direct_inputs_changed)
        for spin in self._delivery_spins.values():
            spin.valueChanged.connect(self._mark_delivery_inputs_changed)
        self._mode_combo.currentIndexChanged.connect(self._mark_inputs_changed)
        self._model_combo.currentIndexChanged.connect(self._mark_inputs_changed)
        self._direction_convention_combo.currentIndexChanged.connect(
            self._mark_direction_changed
        )
        self.wind_controls.enabled_check.toggled.connect(self._mark_inputs_changed)
        self.wind_controls.speed_spin.valueChanged.connect(
            self._mark_wind_inputs_changed
        )
        self.wind_controls.bearing_spin.valueChanged.connect(
            self._mark_wind_inputs_changed
        )

    def _mark_direct_inputs_changed(self) -> None:
        if self.mode() == ENTRY_MODES[0]:
            self._mark_inputs_changed()

    def _mark_delivery_inputs_changed(self) -> None:
        if self.mode() == ENTRY_MODES[1]:
            self._mark_inputs_changed()

    def _mark_wind_inputs_changed(self) -> None:
        if self.wind_controls.enabled_check.isChecked():
            self._mark_inputs_changed()

    def _mark_direction_changed(self) -> None:
        if self.mode() == ENTRY_MODES[0]:
            self._mark_inputs_changed()

    def _show_sample_error(self, message: str, restoration_failed: bool) -> None:
        self._show_error(
            RuntimeError(message),
            origin="selection",
            restoration_failed=restoration_failed,
        )

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
