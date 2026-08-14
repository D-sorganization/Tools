"""Atomic execution/publication mixin for the PyQt Flight Explorer."""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from rate_of_closure.flight_accepted_study import (
    AcceptedFlightStudy,
    FlightStudyContext,
    build_accepted_flight_study,
)
from rate_of_closure.simulation import (
    FlightExploration,
    explore_with_optional_wind,
    launch_from_delivery,
    launch_from_direct,
)
from rate_of_closure.ui.pyqt6.flight_explorer_controls import (
    DISTANCE_ROWS,
    ENTRY_MODES,
    EXPLORER_ROWS,
)
from rate_of_closure.ui.pyqt6.flight_view_bundle import FlightViewRestorationError
from rate_of_closure.units import format_distance_m
from shared.python.swing_sim.flight import LaunchDirectionConvention
from shared.python.swing_sim.impact import DeliveryParameters

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QLabel

    from rate_of_closure.ui.pyqt6.flight_playback_controls import FlightPlaybackPanel
    from rate_of_closure.ui.pyqt6.flight_view import FlightView
    from rate_of_closure.ui.pyqt6.flight_wind_controls import FlightWindControls
    from rate_of_closure.ui.pyqt6.result_row import ResultRow
    from rate_of_closure.ui.pyqt6.spatial_target_workflow import SpatialTargetWorkflow

logger = logging.getLogger(__name__)


class FlightExplorerRunMixin:
    """Own one accepted bundle and publish it with strong result retention."""

    if TYPE_CHECKING:
        _accepted: AcceptedFlightStudy | None
        _generation: int
        _speed_spin: QDoubleSpinBox
        _speed_unit: str
        _direct_spins: dict[str, QDoubleSpinBox]
        _delivery_spins: dict[str, QDoubleSpinBox]
        _direction_convention_combo: QComboBox
        _model_combo: QComboBox
        _rows: dict[str, ResultRow]
        _flight_view: FlightView
        _flight_panel: FlightPlaybackPanel
        _target_workflow: SpatialTargetWorkflow
        _context_status: QLabel
        _sample_status: QLabel
        _error_status: QLabel
        _error_origin: str | None
        wind_controls: FlightWindControls

        def mode(self) -> str: ...

        def speed_mps(self) -> float: ...

        def speed_mph(self) -> float: ...

    def _candidate_context(self) -> tuple[object, FlightStudyContext]:
        scenario = self.wind_controls.optional_scenario()
        model_name = self._model_combo.currentText()
        direction = self._direction_convention_combo.currentData()
        speed_mph = self.speed_mph()
        if self.mode() == ENTRY_MODES[0]:
            values = tuple(
                (key, self._direct_spins[key].value())
                for key in (
                    "launch_angle_deg",
                    "launch_direction_deg",
                    "spin_rpm",
                    "spin_axis_tilt_deg",
                )
            )
            inputs = (("ball_speed_mph", speed_mph), *values)
            launch = launch_from_direct(
                *(value for _key, value in inputs), direction_convention=direction
            )
            mode = "direct"
        else:
            direction = LaunchDirectionConvention.APP_NATIVE
            delivery = DeliveryParameters(
                clubhead_speed_mps=self.speed_mps(),
                **{key: spin.value() for key, spin in self._delivery_spins.items()},
            )
            launch = launch_from_delivery(delivery)
            inputs = (
                ("clubhead_speed_mps", self.speed_mps()),
                ("club_path_deg", delivery.club_path_deg),
                ("face_angle_deg", delivery.face_angle_deg),
                ("attack_angle_deg", delivery.attack_angle_deg),
                ("dynamic_loft_deg", delivery.dynamic_loft_deg),
                ("impact_offset_toe_mm", delivery.impact_offset_toe_mm),
                ("impact_offset_high_mm", delivery.impact_offset_high_mm),
                ("lie_deg", delivery.lie_deg),
            )
            mode = "delivery"
        resolved = replace(launch, wind_speed=0.0, wind_scenario=scenario)
        return launch, FlightStudyContext(
            mode, inputs, direction, model_name, scenario, resolved
        )

    def run_now(self) -> FlightExploration | None:
        """Build and atomically publish one complete accepted flight."""
        prior = self._accepted
        prior_selection = self._flight_view.selected_raw_index()
        prior_playback_time = self._flight_panel.controls.current_time_s()
        prior_rows = {key: row.value_label.text() for key, row in self._rows.items()}
        prior_deltas = self.wind_controls.delta_texts()
        prior_context = self._context_status.text()
        prior_sample = self._sample_status.text()
        prior_error = self._error_status.text()
        prior_error_origin = self._error_origin
        prior_target = self._target_workflow.publication_snapshot()
        target_published = False
        view_published = False
        try:
            launch, context = self._candidate_context()
            exploration, comparison = explore_with_optional_wind(
                launch, context.wind_scenario, context.model_name
            )
            candidate = build_accepted_flight_study(
                self._generation + 1, context, exploration, comparison
            )
            row_texts = self._row_texts(candidate.exploration)
            calm = candidate.calm_comparison
            self._target_workflow.set_trajectory(candidate.exploration.positions)
            target_published = True
            self._flight_view.adopt_sample_bundle(
                candidate.plan,
                None if calm is None else calm.times,
                None if calm is None else calm.positions,
            )
            view_published = True
            self.wind_controls.set_comparison(candidate.comparison)
            for key, text in row_texts.items():
                self._rows[key].value_label.setText(text)
            self._context_status.setText(
                f"Displayed flight: {candidate.context.label()}"
            )
            self._sample_status.setText(
                "Select the current primary trajectory; calm ghost is comparison-only."
            )
            self._error_status.clear()
            self._error_origin = None
            self._accepted = candidate
            self._generation = candidate.generation
            self._exploration = candidate.exploration
            self.wind_comparison = candidate.comparison
        except Exception as exc:
            restoration_failed = self._restore_publication(
                prior,
                prior_selection,
                prior_playback_time,
                prior_rows,
                prior_deltas,
                prior_context,
                prior_sample,
                prior_error,
                prior_error_origin,
                prior_target,
                target_published=target_published,
                view_published=view_published,
            )
            logger.warning("flight exploration failed: %s", exc)
            self._show_error(exc, restoration_failed=restoration_failed)
            return None
        return candidate.exploration

    def _restore_publication(
        self,
        prior: AcceptedFlightStudy | None,
        prior_selection: int | None,
        prior_playback_time: float,
        prior_rows: dict[str, str],
        prior_deltas: dict[str, str],
        prior_context: str,
        prior_sample: str,
        prior_error: str,
        prior_error_origin: str | None,
        prior_target: tuple[np.ndarray, str],
        *,
        target_published: bool,
        view_published: bool,
    ) -> bool:
        """Restore every prior publication surface; report any failed seam."""
        restoration_failed = False
        if view_published:
            try:
                if prior is None:
                    self._flight_view.clear_sample_bundle()
                else:
                    calm = prior.calm_comparison
                    self._flight_view.adopt_sample_bundle(
                        prior.plan,
                        None if calm is None else calm.times,
                        None if calm is None else calm.positions,
                        selected_raw_index=prior_selection,
                        playback_time_s=prior_playback_time,
                    )
                    self._flight_panel.controls.jump_to_time(prior_playback_time)
            except Exception:
                restoration_failed = True
                logger.exception("flight view publication rollback failed")
                try:
                    calm = None if prior is None else prior.calm_comparison
                    self._flight_view.force_sample_bundle_authority(
                        None if prior is None else prior.plan,
                        None if calm is None else calm.times,
                        None if calm is None else calm.positions,
                        selected_raw_index=prior_selection,
                        playback_time_s=prior_playback_time,
                    )
                    self._flight_panel.controls.jump_to_time(prior_playback_time)
                except Exception:
                    logger.exception("flight view authority force-restore failed")
        if target_published:
            try:
                self._target_workflow.restore_publication_snapshot(prior_target)
            except Exception:
                restoration_failed = True
                logger.exception("flight target workflow rollback failed")
        try:
            self.wind_controls.restore_delta_texts(prior_deltas)
            for key, text in prior_rows.items():
                self._rows[key].value_label.setText(text)
            self._context_status.setText(prior_context)
            self._sample_status.setText(prior_sample)
            self._error_status.setText(prior_error)
            self._error_origin = prior_error_origin
        except Exception:
            restoration_failed = True
            logger.exception("flight presentation rollback failed")
        return restoration_failed

    def _row_texts(self, exploration: FlightExploration) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, _label, unit in EXPLORER_ROWS:
            value = exploration.metrics[key]
            if not math.isfinite(value):
                raise ValueError("accepted flight row must be finite")
            if key in DISTANCE_ROWS:
                text = (
                    f"+{format_distance_m(value)}"
                    if value >= 0
                    else f"-{format_distance_m(-value)}"
                )
            else:
                text = f"{value:+.1f}{unit}"
            result[key] = text
        return result

    def refresh_units(self) -> None:
        """Presentation-only row refresh preserves accepted identity and warning."""
        if self._accepted is None:
            return
        for key, text in self._row_texts(self._accepted.exploration).items():
            self._rows[key].value_label.setText(text)

    def _on_sample_selected(self, raw_index: int) -> None:
        if self._error_origin == "selection":
            self._error_status.clear()
            self._error_origin = None
        accepted = self._accepted
        if accepted is None or raw_index < 0:
            self._sample_status.setText(
                "Select the current primary trajectory; calm ghost is comparison-only."
            )
            return
        sample = accepted.plan.raw_sample(raw_index)
        self._flight_panel.controls.jump_to_time(sample.time_s)
        sample_number = raw_index + 1
        self._sample_status.setText(
            f"Current primary flight, source sample {sample_number}/"
            f"{accepted.plan.raw_count}; "
            f"t {sample.time_s:.3f} s; downrange {sample.downrange_m:.3f} m; "
            f"height {sample.height_m:.3f} m; right {sample.right_m:.3f} m; "
            f"{sample.phase}."
        )

    def _mark_inputs_changed(self, *_args: object) -> None:
        if self._accepted is not None:
            self._context_status.setText(
                f"Prior result — inputs changed: {self._accepted.context.label()}"
            )

    def _show_error(
        self,
        error: Exception,
        *,
        origin: str = "scientific",
        restoration_failed: bool = False,
    ) -> None:
        retained = (
            "Prior accepted authority is retained, but plot restoration failed; "
            "the image may be stale or unavailable."
            if restoration_failed or isinstance(error, FlightViewRestorationError)
            else "The prior accepted flight remains displayed."
            if self._accepted is not None
            else "No accepted flight is available."
        )
        sanitized = "".join(
            " " if ord(character) < 32 or 127 <= ord(character) <= 159 else character
            for character in str(error)
        )
        text = " ".join(sanitized.split()) or "Flight computation failed"
        self._error_status.setText(f"{text[: 238 - len(retained)]}. {retained}")
        self._error_origin = origin
