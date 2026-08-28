"""Orbitable 3-D putt playback for the Putting tab (#4800 P6 / P8).

The putting half of Amendment 1's one playback architecture: the ball
rolling its break line across the 3-D green, replayed on the shared
timeline. Two rules from P8 hold here verbatim.

**Frames come from the recorded samples, never from re-simulation.**
:func:`putt_playback_trajectory` lifts the retained
:class:`~shared.python.swing_sim.putting.PuttResult` samples to the
existing :class:`~rate_of_closure.simulation.flight_playback.TimedTrajectory`
— the same sample-to-frame contract the flight views use — by reading
each sample's elevation off the same
:class:`~shared.python.swing_sim.putting.GreenSurface` the integrator
ran on. Nothing is re-integrated and nothing is resampled.

**Transport and camera state live elsewhere.** This widget takes a
physical time through :meth:`PuttPlaybackView.set_time` and owns no
timer, no speed, and no scrub: P8's subject-neutral
``PlaybackTransportControls`` drives it with "Putt" wording and
Strike/Finish events. The camera is the Matplotlib 3-D axes' own orbit
(drag to rotate), so named cameras arrive with #4571's
``CameraViewportMixin`` without this module owning any camera state.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers "3d" projection
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from rate_of_closure.simulation.flight_playback import TimedTrajectory
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)
from shared.python.swing_sim.impact import GOLF_BALL_RADIUS_M
from shared.python.swing_sim.putting import (
    HOLE_RADIUS_M,
    GreenSurface,
    PuttResult,
)

__all__ = ["PuttPlaybackView", "putt_playback_trajectory"]

#: Nodes per axis in the drawn green mesh (display only; the physics
#: samples the surface analytically at every RK4 stage).
_MESH_NODES = 24

#: Padding around the path/hole bounding box in the drawn green [m].
_MESH_PAD_M = 0.35


def putt_playback_trajectory(
    result: PuttResult, surface: GreenSurface
) -> TimedTrajectory:
    """Lift one integrated putt to the shared playback trajectory.

    Args:
        result: The integrated putt whose retained samples are replayed.
        surface: The exact green the putt was integrated on; elevations
            are read from it so the ball rides the drawn surface.

    Returns:
        A :class:`TimedTrajectory` of ``(x, y, z)`` ball-centre
        positions [m] at the recorded sample times, where ``x`` is the
        target line, ``y`` is lateral (left positive) and ``z`` is
        elevation.

    Raises:
        TypeError: If ``result`` is not a :class:`PuttResult`.
        ValueError: If the retained samples are not a valid timeline.
    """
    if not isinstance(result, PuttResult):
        raise TypeError("result must be a PuttResult")
    heights = [
        surface.height_m(x_m, y_m) + GOLF_BALL_RADIUS_M
        for x_m, y_m in zip(result.path_x_m, result.path_y_m, strict=True)
    ]
    positions = np.column_stack(
        (
            np.asarray(result.path_x_m, dtype=float),
            np.asarray(result.path_y_m, dtype=float),
            np.asarray(heights, dtype=float),
        )
    )
    return TimedTrajectory(
        times_s=np.asarray(result.times_s, dtype=float), positions_m=positions
    )


class PuttPlaybackView(QWidget):
    """Rotatable 3-D green with the break line and a time-driven ball."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5.0, 3.4), layout="constrained")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._canvas.setAccessibleName("Rotatable 3D putt playback")
        self._canvas.setAccessibleDescription(
            "Drag to orbit the green. The ball follows the recorded "
            "trajectory samples at the transport's physical time."
        )
        self._canvas.setToolTip(
            "3-D green, break line, and hole. Drag to rotate the "
            "camera; the transport below scrubs physical time."
        )
        self._canvas.setMinimumHeight(220)
        self._status = QLabel("No putt is loaded for playback.")
        self._status.setAccessibleName("Putt playback position")
        self._status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._status)
        self._trajectory: TimedTrajectory | None = None
        self._ball: object | None = None
        self._holed = False

    # ── seams ───────────────────────────────────────────────────────
    def canvas(self) -> FigureCanvas:
        """The sole focusable visual surface (probe seam)."""
        return self._canvas

    def status_text(self) -> str:
        """The visible ball position at the current playback time."""
        return str(self._status.text())

    def duration_s(self) -> float:
        """Physical duration of the loaded putt [s]; 0 when none is."""
        return 0.0 if self._trajectory is None else self._trajectory.duration_s

    def event_times_s(self) -> tuple[float, float]:
        """Strike and finish timestamps for the transport's jumps [s]."""
        return (0.0, self.duration_s())

    def trajectory(self) -> TimedTrajectory | None:
        """The adopted playback trajectory (test seam)."""
        return self._trajectory

    # ── behaviour ───────────────────────────────────────────────────
    def set_putt(
        self,
        result: PuttResult,
        surface: GreenSurface,
        *,
        hole_distance_m: float,
    ) -> None:
        """Adopt one integrated putt and draw its static scene."""
        self._trajectory = putt_playback_trajectory(result, surface)
        self._holed = bool(result.holed)
        self._draw_scene(surface, hole_distance_m, result.skid_end_index)
        self.set_time(0.0)

    def clear(self) -> None:
        """Drop the scene without disturbing an accepted result elsewhere."""
        self._trajectory = None
        self._ball = None
        self._figure.clear()
        self._status.setText("No putt is loaded for playback.")
        self._canvas.draw_idle()

    def set_time(self, time_s: float) -> None:
        """Move the ball to the recorded frame at one physical time [s]."""
        if self._trajectory is None or self._ball is None:
            return
        frame = self._trajectory.frame_at(time_s)
        x_m, y_m, z_m = (float(value) for value in frame.position_m)
        self._ball.set_data_3d([x_m], [y_m], [z_m])  # type: ignore[attr-defined]
        outcome = "holed" if self._holed and frame.is_landing else "in play"
        self._status.setText(
            f"t {frame.time_s:.3f} s of {self._trajectory.duration_s:.3f} s; "
            f"x {x_m:.3f} m; y {y_m:.3f} m; elevation {z_m:.3f} m; {outcome}."
        )
        self._canvas.draw_idle()

    # ── rendering ───────────────────────────────────────────────────
    def _draw_scene(
        self, surface: GreenSurface, hole_distance_m: float, skid_end_index: int
    ) -> None:
        assert self._trajectory is not None
        positions = self._trajectory.positions_m
        self._figure.clear()
        axes = self._figure.add_subplot(projection="3d")
        self._draw_green_mesh(axes, surface, positions, hole_distance_m)
        split = max(1, min(skid_end_index + 1, len(positions)))
        axes.plot(
            positions[:split, 0],
            positions[:split, 1],
            positions[:split, 2],
            color="tab:orange",
            linewidth=2.0,
            label="Skid",
        )
        axes.plot(
            positions[split - 1 :, 0],
            positions[split - 1 :, 1],
            positions[split - 1 :, 2],
            color="tab:green",
            linewidth=2.0,
            label="Pure roll",
        )
        self._draw_hole(axes, surface, hole_distance_m)
        (self._ball,) = axes.plot(
            [positions[0, 0]],
            [positions[0, 1]],
            [positions[0, 2]],
            marker="o",
            markersize=6,
            color="#eab308",
            linestyle="none",
            label="Ball",
        )
        axes.set_xlabel("Along target line [m]")
        axes.set_ylabel("Lateral [m] (left +)")
        axes.set_zlabel("Elevation [m]")
        axes.set_title("3-D green playback")
        axes.legend(loc="upper left", fontsize=7)
        axes.view_init(elev=32.0, azim=-118.0)
        self._canvas.draw_idle()

    def _draw_green_mesh(
        self,
        axes: object,
        surface: GreenSurface,
        positions: np.ndarray,
        hole_distance_m: float,
    ) -> None:
        x_values = np.linspace(
            min(float(positions[:, 0].min()), 0.0) - _MESH_PAD_M,
            max(float(positions[:, 0].max()), hole_distance_m) + _MESH_PAD_M,
            _MESH_NODES,
        )
        y_values = np.linspace(
            float(positions[:, 1].min()) - _MESH_PAD_M,
            float(positions[:, 1].max()) + _MESH_PAD_M,
            _MESH_NODES,
        )
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        grid_z = np.array(
            [
                [surface.height_m(float(x_m), float(y_m)) for x_m in x_values]
                for y_m in y_values
            ]
        )
        axes.plot_surface(  # type: ignore[attr-defined]
            grid_x,
            grid_y,
            grid_z,
            color="#86efac",
            alpha=0.35,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )

    def _draw_hole(
        self, axes: object, surface: GreenSurface, hole_distance_m: float
    ) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 48)
        rim_x = hole_distance_m + HOLE_RADIUS_M * np.cos(angles)
        rim_y = HOLE_RADIUS_M * np.sin(angles)
        rim_z = np.array(
            [
                surface.height_m(float(x_m), float(y_m))
                for x_m, y_m in zip(rim_x, rim_y, strict=True)
            ]
        )
        axes.plot(  # type: ignore[attr-defined]
            rim_x, rim_y, rim_z, color="black", linewidth=1.6, label="Hole"
        )
