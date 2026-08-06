"""Dedicated ball-flight viewer — flight-scale display (tens of metres).

One of the three scale-separated viewers (epic #4120, V2): a side
profile (height vs carry), a top-down view (lateral vs carry), and the
3D trajectory polyline, all auto-scaled to the flight regime, with the
landing point annotated. Each panel and annotation has its own display
checkbox with sourced guidance. The view accepts either a full
:class:`~rate_of_closure.simulation.SimulationRun` or a bare
trajectory, so the standalone Flight Explorer reuses it with no swing.

Colors come from the shared UpstreamDrift theme palette
(``get_chart_color``); no app colors are hard-coded.
"""

from __future__ import annotations

import logging

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from rate_of_closure.simulation import SimulationRun
from rate_of_closure.simulation.flight_playback import TimedTrajectory
from rate_of_closure.simulation.targets import TargetRegion
from rate_of_closure.ui.course import CourseLayout
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)
from rate_of_closure.ui.pyqt6.flight_playback_rendering import FlightPlaybackArtists
from rate_of_closure.ui.pyqt6.flight_view_axes import distance_axis
from rate_of_closure.ui.pyqt6.flight_view_panels import FlightViewPanelsMixin
from rate_of_closure.units import FIELD_GUIDANCE

logger = logging.getLogger(__name__)

__all__ = ["FlightView", "distance_axis"]

#: Minimum plotted extents so degenerate flights stay readable.
_MIN_CARRY_M = 10.0
_MIN_HEIGHT_M = 5.0
_MIN_LATERAL_M = 5.0

#: (checkbox attribute, label, FIELD_GUIDANCE key, default) per display
#: parameter, in bar order.
_DISPLAY_PARAMS: tuple[tuple[str, str, str, bool], ...] = (
    ("side", "Side Profile", "flight_side_visible", True),
    ("top", "Top-Down", "flight_top_visible", True),
    ("three_d", "3D Trajectory", "flight_3d_visible", True),
    ("landing", "Landing Point", "flight_landing_visible", True),
    ("apex", "Apex", "flight_apex_visible", True),
    ("course", "Course Elements", "course_visible", True),
)


# Repository-wide duplicate-module protection skips local import traversal, so
# mypy cannot resolve the extracted mixin at this import boundary.
class FlightView(FlightViewPanelsMixin, QWidget):  # type: ignore[misc]
    """Flight-scale trajectory viewer: side + top-down 2D panels + 3D."""

    timelineChanged = pyqtSignal(float)  # noqa: N815 - Qt signal convention

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(7, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)

        self._positions: np.ndarray = np.zeros((0, 3))
        self.comparison_positions: np.ndarray = np.zeros((0, 3))
        self._timed_trajectory: TimedTrajectory | None = None
        self._comparison_timed: TimedTrajectory | None = None
        self._playback_time_s = 0.0
        self._playback_artists = FlightPlaybackArtists()
        self._run: SimulationRun | None = None
        self._checks: dict[str, QCheckBox] = {}
        self._course_layout = CourseLayout()
        self._target_region: TargetRegion | None = None
        # (carry, lateral) landing scatter [m] from the Variation engine.
        self._scatter: tuple[np.ndarray, np.ndarray] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._build_param_bar())
        layout.addWidget(self._canvas)
        self._draw()

    def _build_param_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 4, 4, 0)
        for attr, label, guidance_key, default in _DISPLAY_PARAMS:
            check = QCheckBox(label)
            check.setChecked(default)
            check.setToolTip(FIELD_GUIDANCE[guidance_key])
            check.toggled.connect(lambda _checked: self._draw())
            self._checks[attr] = check
            bar.addWidget(check)
        bar.addStretch(1)
        return bar

    # ── public API ──────────────────────────────────────────────────
    def set_run(self, run: SimulationRun | None) -> None:
        """Adopt the flight trajectory of a full simulation run."""
        self._run = run
        self.comparison_positions = np.zeros((0, 3))
        self._comparison_timed = None
        self._positions = (
            np.zeros((0, 3)) if run is None else run.flight_positions.copy()
        )
        self._timed_trajectory = (
            None
            if run is None or not len(run.flight_times)
            else TimedTrajectory(run.flight_times, run.flight_positions)
        )
        self._reset_playback()
        self._draw()

    def set_trajectory(self, positions: np.ndarray | None) -> None:
        """Adopt an (N, 3) app-frame trajectory (or clear with ``None``).

        App frame: x downrange along the target line [m], y up [m],
        z right of target [m].
        """
        self._run = None
        self.comparison_positions = np.zeros((0, 3))
        self._timed_trajectory = None
        self._comparison_timed = None
        self._positions = (
            np.zeros((0, 3)) if positions is None else np.asarray(positions, float)
        )
        self._reset_playback()
        self._draw()

    def set_timed_trajectory(
        self, times_s: np.ndarray, positions_m: np.ndarray
    ) -> None:
        """Adopt a solver-timestamped app-frame trajectory for playback."""
        self._run = None
        self._timed_trajectory = TimedTrajectory(times_s, positions_m)
        self._positions = self._timed_trajectory.positions_m
        self._reset_playback()
        self._draw()

    def set_comparison_trajectory(self, positions: np.ndarray | None) -> None:
        """Overlay an optional common-input no-wind trajectory."""
        self._comparison_timed = None
        self.comparison_positions = (
            np.zeros((0, 3)) if positions is None else np.asarray(positions, float)
        )
        self._draw()

    def set_comparison_timed_trajectory(
        self, times_s: np.ndarray | None, positions_m: np.ndarray | None
    ) -> None:
        """Overlay an optional timestamped comparison trajectory."""
        self._comparison_timed = (
            None
            if times_s is None or positions_m is None
            else TimedTrajectory(times_s, positions_m)
        )
        self.comparison_positions = (
            np.zeros((0, 3))
            if self._comparison_timed is None
            else self._comparison_timed.positions_m
        )
        self._draw()

    def set_playback_time(self, time_s: float) -> None:
        """Move playback markers without rebuilding axes or camera state."""
        if self._timed_trajectory is None:
            return
        frame = self._timed_trajectory.frame_at(time_s)
        self._playback_time_s = frame.time_s
        self._playback_artists.update(frame.position_m)
        self._canvas.draw_idle()

    def playback_duration_s(self) -> float:
        """Current solver trajectory duration [s], or zero when unavailable."""
        return (
            0.0 if self._timed_trajectory is None else self._timed_trajectory.duration_s
        )

    def _reset_playback(self) -> None:
        self._playback_time_s = 0.0
        self.timelineChanged.emit(self.playback_duration_s())

    def trajectory(self) -> np.ndarray:
        """The (N, 3) app-frame trajectory currently rendered."""
        return self._positions

    def display_check(self, name: str) -> QCheckBox:
        """The display-parameter checkbox for ``name`` (test seam)."""
        return self._checks[name]

    def course_layout(self) -> CourseLayout:
        """The course furniture layout rendered behind the flight."""
        return self._course_layout

    def set_course_layout(self, layout: CourseLayout) -> None:
        """Adopt a course layout (H7b target edits drive this) and redraw."""
        self._course_layout = layout
        self._draw()

    def set_target_region(self, region: TargetRegion | None) -> None:
        """Show (or clear) the target-region boundary in the top-down view."""
        self._target_region = region
        self._draw()

    def target_region(self) -> TargetRegion | None:
        """The target region currently overlaid, if any."""
        return self._target_region

    def set_landing_scatter(
        self, carries_m: np.ndarray | None, laterals_m: np.ndarray | None = None
    ) -> None:
        """Overlay a Variation landing scatter (or clear with ``None``).

        #4125 H7b tie-in: when a target region is set, the top-down
        title reports the share of shots holding the target — the
        headline Monte-Carlo output.
        """
        if carries_m is None or laterals_m is None:
            self._scatter = None
        else:
            self._scatter = (
                np.asarray(carries_m, float),
                np.asarray(laterals_m, float),
            )
        self._draw()

    def extents_m(self) -> tuple[float, float, float]:
        """(carry, height, lateral) plot extents [m] — flight regime."""
        pos = self._positions
        if not len(pos):
            return (_MIN_CARRY_M, _MIN_HEIGHT_M, _MIN_LATERAL_M)
        all_positions = (
            np.vstack((pos, self.comparison_positions))
            if len(self.comparison_positions)
            else pos
        )
        carry = max(_MIN_CARRY_M, float(np.max(all_positions[:, 0])) * 1.05)
        lateral = max(_MIN_LATERAL_M, float(np.max(np.abs(all_positions[:, 2]))) * 1.3)
        if self._scatter is not None and len(self._scatter[0]):
            carries, laterals = self._scatter
            finite = np.isfinite(carries) & np.isfinite(laterals)
            if np.any(finite):
                carry = max(carry, float(np.max(carries[finite])) * 1.05)
                lateral = max(lateral, float(np.max(np.abs(laterals[finite]))) * 1.1)
        return (
            carry,
            max(_MIN_HEIGHT_M, float(np.max(all_positions[:, 1])) * 1.2),
            lateral,
        )

    # ── drawing ─────────────────────────────────────────────────────
    def _draw(self) -> None:
        self._figure.clear()
        pos = self._positions
        frame = (
            None
            if self._timed_trajectory is None
            else self._timed_trajectory.frame_at(self._playback_time_s)
        )
        self._playback_artists.reset(None if frame is None else frame.position_m)
        panels = [
            name
            for name in ("side", "top", "three_d")
            if self._checks[name].isChecked()
        ]
        if not len(pos) or not panels:
            axes = self._figure.add_subplot(111)
            axes.set_xticks([])
            axes.set_yticks([])
            if not len(pos) and self._run is not None:
                title = (
                    "No Flight — fixed-ball contact was missed; "
                    "the swing remains available"
                )
            elif not len(pos):
                title = "Run a flight to populate the view"
            else:
                title = "Enable a panel to display the flight"
            axes.set_title(title)
            self._canvas.draw_idle()
            return

        want_3d = "three_d" in panels
        left = [name for name in panels if name != "three_d"]
        grid = self._figure.add_gridspec(
            max(len(left), 1), 2 if want_3d and left else 1
        )
        extents = self.extents_m()
        for row, name in enumerate(left):
            axes = self._figure.add_subplot(grid[row, 0])
            if name == "side":
                self._draw_side(axes, pos, extents)
            else:
                self._draw_top(axes, pos, extents)
        if want_3d:
            spec = grid[:, 1] if left else grid[:, 0]
            axes_3d = self._figure.add_subplot(spec, projection="3d")
            self._draw_3d(axes_3d, pos, extents)
        self._canvas.draw_idle()
