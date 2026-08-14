"""Atomic primary/comparison/sample-plan adoption for :mod:`flight_view`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rate_of_closure.flight_sample_inspector import FlightSamplePlan
from rate_of_closure.simulation.flight_playback import TimedTrajectory

if TYPE_CHECKING:
    from rate_of_closure.simulation import SimulationRun
    from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas

logger = logging.getLogger(__name__)


class FlightViewRestorationError(RuntimeError):
    """Accepted authority rolled back, but prior pixels could not be restored."""


class FlightViewBundleMixin:
    """Stage a complete inspector bundle and publish one rendered frame."""

    if TYPE_CHECKING:
        _run: SimulationRun | None
        _timed_trajectory: TimedTrajectory | None
        _comparison_timed: TimedTrajectory | None
        _positions: np.ndarray
        comparison_positions: np.ndarray
        _sample_plan: FlightSamplePlan | None
        _selected_raw_index: int | None
        _playback_time_s: float
        _canvas: LifecycleSafeFigureCanvas
        timelineChanged: Any

        def _draw(self, *, sync: bool = False) -> None: ...

        def set_sample_plan(self, plan: FlightSamplePlan | None) -> None: ...

        def playback_duration_s(self) -> float: ...

        def playback_apex_time_s(self) -> float: ...

    def adopt_sample_bundle(
        self,
        plan: FlightSamplePlan,
        comparison_times: np.ndarray | None,
        comparison_positions: np.ndarray | None,
        *,
        selected_raw_index: int | None = None,
        playback_time_s: float = 0.0,
    ) -> None:
        """Publish primary, calm ghost, plan, and playback once or roll back."""
        primary = TimedTrajectory(
            np.asarray(plan.series.times_s), np.asarray(plan.series.positions_m)
        )
        if (comparison_times is None) != (comparison_positions is None):
            raise ValueError("comparison times and positions must be provided together")
        comparison = (
            None
            if comparison_times is None or comparison_positions is None
            else TimedTrajectory(comparison_times, comparison_positions)
        )
        if selected_raw_index is not None:
            plan.raw_sample(selected_raw_index)
        self._publish_sample_state(
            primary,
            comparison,
            plan,
            selected_raw_index,
            playback_time_s,
        )

    def clear_sample_bundle(self) -> None:
        """Atomically publish the honest no-accepted-flight state."""
        self._publish_sample_state(None, None, None, None, 0.0)

    def force_sample_bundle_authority(
        self,
        plan: FlightSamplePlan | None,
        comparison_times: np.ndarray | None,
        comparison_positions: np.ndarray | None,
        *,
        selected_raw_index: int | None = None,
        playback_time_s: float = 0.0,
    ) -> None:
        """Restore trusted prior authority when its pixels cannot be repainted."""
        if plan is None:
            if comparison_times is not None or comparison_positions is not None:
                raise ValueError("an empty authority cannot carry a comparison")
            primary = None
            comparison = None
        else:
            primary = TimedTrajectory(
                np.asarray(plan.series.times_s), np.asarray(plan.series.positions_m)
            )
            if (comparison_times is None) != (comparison_positions is None):
                raise ValueError(
                    "comparison times and positions must be provided together"
                )
            comparison = (
                None
                if comparison_times is None or comparison_positions is None
                else TimedTrajectory(comparison_times, comparison_positions)
            )
            if selected_raw_index is not None:
                plan.raw_sample(selected_raw_index)
        self._install_sample_state(
            primary,
            comparison,
            plan,
            selected_raw_index,
            playback_time_s,
        )
        self._canvas.pause_idle_draws()
        self.timelineChanged.emit(
            self.playback_duration_s(), self.playback_apex_time_s()
        )

    def _publish_sample_state(
        self,
        primary: TimedTrajectory | None,
        comparison: TimedTrajectory | None,
        plan: FlightSamplePlan | None,
        selected_raw_index: int | None,
        playback_time_s: float,
    ) -> None:
        previous = (
            self._run,
            self._timed_trajectory,
            self._comparison_timed,
            self._positions,
            self.comparison_positions,
            self._sample_plan,
            self._selected_raw_index,
            self._playback_time_s,
        )
        self._install_sample_state(
            primary,
            comparison,
            plan,
            selected_raw_index,
            playback_time_s,
        )
        try:
            self._draw(sync=True)
        except Exception as publication_error:
            (
                self._run,
                self._timed_trajectory,
                self._comparison_timed,
                self._positions,
                self.comparison_positions,
                self._sample_plan,
                self._selected_raw_index,
                self._playback_time_s,
            ) = previous
            try:
                self._draw(sync=True)
            except Exception:
                logger.exception("flight view rollback render failed")
                self._canvas.pause_idle_draws()
                raise FlightViewRestorationError(
                    "prior accepted authority was retained, but plot restoration "
                    "failed; the image may be stale or unavailable"
                ) from publication_error
            self._canvas.resume_idle_draws()
            raise
        self._canvas.resume_idle_draws()
        self.timelineChanged.emit(
            self.playback_duration_s(), self.playback_apex_time_s()
        )

    def _install_sample_state(
        self,
        primary: TimedTrajectory | None,
        comparison: TimedTrajectory | None,
        plan: FlightSamplePlan | None,
        selected_raw_index: int | None,
        playback_time_s: float,
    ) -> None:
        """Install already validated state without attempting a render."""
        self._run = None
        self._timed_trajectory = primary
        self._comparison_timed = comparison
        self._positions = np.zeros((0, 3)) if primary is None else primary.positions_m
        self.comparison_positions = (
            np.zeros((0, 3)) if comparison is None else comparison.positions_m
        )
        self.set_sample_plan(plan)
        self._selected_raw_index = selected_raw_index
        self._playback_time_s = (
            0.0 if primary is None else primary.frame_at(playback_time_s).time_s
        )


__all__ = ["FlightViewBundleMixin", "FlightViewRestorationError"]
