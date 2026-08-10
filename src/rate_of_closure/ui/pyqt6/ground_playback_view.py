"""Locked-scale interactive 3D view for imported ground trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from rate_of_closure.simulation.ground_playback import GroundPlaybackTimeline
from rate_of_closure.simulation.ground_playback_workspace import (
    MAX_CAMERA_ZOOM,
    MIN_CAMERA_ZOOM,
    GroundPlaybackViewState,
)
from rate_of_closure.ui.pyqt6.figure_canvas import LifecycleSafeFigureCanvas

PHASE_COLORS = {
    "impact": "#ef476f",
    "bounce": "#a78bfa",
    "skid": "#f59e0b",
    "roll": "#34d399",
    "rest": "#60a5fa",
}
_MIN_SPAN_M = 0.05


class GroundPlayback3DView(QWidget):
    """Render a phase-colored path while preserving mouse camera control."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(7.0, 5.0), constrained_layout=True)
        self.canvas = LifecycleSafeFigureCanvas(self.figure)
        self.axes: Any = self.figure.add_subplot(111, projection="3d")
        self._timeline: GroundPlaybackTimeline | None = None
        self._ball: Any = None
        self._base_limits: (
            tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None
        ) = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setAccessibleName("Interactive ground trajectory in three dimensions")

    def set_timeline(self, timeline: GroundPlaybackTimeline) -> None:
        """Replace the static trajectory and auto-fit the physical frame."""
        self._timeline = timeline
        self.axes.clear()
        self._draw_phase_segments(timeline)
        self._draw_reference_markers(timeline)
        self._configure_axes(timeline)
        start = timeline.carry_position_m
        self._ball = self.axes.scatter(
            [start[0]], [start[2]], [start[1]], s=80, color="#f6c344", edgecolor="black"
        )
        self.canvas.draw_idle()

    def set_position(self, position_m: tuple[float, float, float]) -> None:
        """Move the playback ball without rebuilding axes or camera state."""
        if self._ball is None:
            return
        self._ball._offsets3d = ([position_m[0]], [position_m[2]], [position_m[1]])
        self.canvas.draw_idle()

    def reset_view(self) -> None:
        """Restore the documented downrange/vertical/side camera."""
        self.axes.view_init(elev=22.0, azim=-62.0)
        if self._timeline is not None:
            self._fit_limits(self._timeline)
        self.canvas.draw_idle()

    def workspace_view(self) -> GroundPlaybackViewState:
        """Capture the portable orbit orientation and current scale zoom."""
        zoom = 1.0
        if self._base_limits is not None:
            base_span = self._base_limits[0][1] - self._base_limits[0][0]
            current = self.axes.get_xlim3d()
            current_span = current[1] - current[0]
            if current_span > 0.0:
                zoom = min(
                    MAX_CAMERA_ZOOM,
                    max(MIN_CAMERA_ZOOM, base_span / current_span),
                )
        yaw = ((float(self.axes.azim) + 180.0) % 360.0) - 180.0
        return GroundPlaybackViewState(yaw, float(self.axes.elev), zoom)

    def apply_workspace_view(
        self, *, yaw_deg: float, pitch_deg: float, zoom: float
    ) -> None:
        """Apply a validated portable orbit camera without changing result data."""
        state = GroundPlaybackViewState(yaw_deg, pitch_deg, zoom)
        self.axes.view_init(elev=state.pitch_deg, azim=state.yaw_deg)
        if self._base_limits is not None:
            self._apply_zoom(state.zoom)
        self.canvas.draw_idle()

    def stop(self) -> None:
        """Cancel deferred Matplotlib redraw work during teardown."""
        self.canvas.cancel_pending_draw()

    def _draw_phase_segments(self, timeline: GroundPlaybackTimeline) -> None:
        points = timeline.result.trajectory
        for index in range(len(points) - 1):
            first, second = points[index : index + 2]
            color = PHASE_COLORS[first.phase.value]
            self.axes.plot(
                [first.position_m[0], second.position_m[0]],
                [first.position_m[2], second.position_m[2]],
                [first.position_m[1], second.position_m[1]],
                color=color,
                linewidth=3.0,
            )
        phases = dict.fromkeys(point.phase.value for point in points)
        for phase in phases:
            self.axes.plot(
                [], [], [], color=PHASE_COLORS[phase], linewidth=3, label=phase.title()
            )

    def _draw_reference_markers(self, timeline: GroundPlaybackTimeline) -> None:
        carry = timeline.carry_position_m
        endpoint = timeline.endpoint_position_m
        self.axes.scatter(
            [carry[0]],
            [carry[2]],
            [carry[1]],
            marker="o",
            s=70,
            facecolors="none",
            edgecolors="#1f77b4",
            linewidths=2,
            label="Carry / first contact",
        )
        self.axes.scatter(
            [endpoint[0]],
            [endpoint[2]],
            [endpoint[1]],
            marker="X",
            s=80,
            color="#111827",
            label=timeline.end_label,
        )

    def _configure_axes(self, timeline: GroundPlaybackTimeline) -> None:
        self.axes.set_xlabel("x downrange [m]", labelpad=5)
        self.axes.set_ylabel("z right [m]", labelpad=5)
        self.axes.set_zlabel("y up [m]", labelpad=5)
        self.axes.tick_params(pad=1, labelsize=8)
        self.axes.set_title("Ground result — target_frame:x_downrange,y_up,z_right")
        self.axes.legend(loc="upper left", fontsize=8)
        self.reset_view()

    def _fit_limits(self, timeline: GroundPlaybackTimeline) -> None:
        positions = np.asarray(
            [point.position_m for point in timeline.result.trajectory]
        )
        low = positions.min(axis=0)
        high = positions.max(axis=0)
        spans = np.maximum(high - low, _MIN_SPAN_M)
        centers = (low + high) / 2.0
        padding = spans * 0.08
        self.axes.set_xlim(
            centers[0] - spans[0] / 2 - padding[0],
            centers[0] + spans[0] / 2 + padding[0],
        )
        self.axes.set_ylim(
            centers[2] - spans[2] / 2 - padding[2],
            centers[2] + spans[2] / 2 + padding[2],
        )
        self.axes.set_zlim(
            centers[1] - spans[1] / 2 - padding[1],
            centers[1] + spans[1] / 2 + padding[1],
        )
        self.axes.set_box_aspect((spans[0], spans[2], spans[1]))
        self._base_limits = (
            tuple(self.axes.get_xlim3d()),
            tuple(self.axes.get_ylim3d()),
            tuple(self.axes.get_zlim3d()),
        )

    def _apply_zoom(self, zoom: float) -> None:
        assert self._base_limits is not None
        setters = (self.axes.set_xlim3d, self.axes.set_ylim3d, self.axes.set_zlim3d)
        for limits, setter in zip(self._base_limits, setters, strict=True):
            center = (limits[0] + limits[1]) / 2.0
            half_span = (limits[1] - limits[0]) / (2.0 * zoom)
            setter(center - half_span, center + half_span)


__all__ = ["GroundPlayback3DView", "PHASE_COLORS"]
