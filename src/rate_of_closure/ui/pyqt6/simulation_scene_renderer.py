"""Rendering-only collaborator for the PyQt6 swing simulation view."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from rate_of_closure.simulation import KineticsSeries
from rate_of_closure.simulation.screw_analysis import (
    JointMotionSeries,
    ScrewMotion,
    analyze_joint_motion,
    analyze_twist,
)
from rate_of_closure.ui.pyqt6.ball_setup_scene import draw_representative_tee
from rate_of_closure.ui.pyqt6.course_scene import draw_course_ground_3d
from rate_of_closure.ui.pyqt6.impact_scene_renderer import draw_impact_scene_3d
from rate_of_closure.ui.pyqt6.kinetics_overlay import overlay_frame
from rate_of_closure.ui.pyqt6.pendulum_scene import draw_pendulum_skeleton
from rate_of_closure.ui.pyqt6.presentation_kinetics import kinetics_for_presentation
from rate_of_closure.ui.pyqt6.screw_overlay import (
    ScrewOverlayRenderer,
    format_screw_readout,
)
from rate_of_closure.ui.pyqt6.wedge_ground_scene import draw_wedge_ground_overlay_3d

__all__ = ["SimulationSceneRenderer", "fallback_joint_ids", "joint_label"]

_BALL_DRAW_RADIUS_M = 0.03
_JOINT_LABELS = {
    "joint.shoulder": "Shoulder Joint",
    "joint.elbow": "Elbow Joint",
    "joint.wrist": "Wrist Joint",
}


def fallback_joint_ids(count: int) -> tuple[str, ...]:
    """Return stable display IDs for articulated sources without torque IDs."""
    canonical = ("joint.shoulder", "joint.elbow", "joint.wrist")
    return tuple(
        canonical[index] if index < len(canonical) else f"joint.{index + 1}"
        for index in range(count)
    )


def joint_label(joint_id: str) -> str:
    """Return an engineering-facing label for one stable joint identifier."""
    return _JOINT_LABELS.get(
        joint_id, joint_id.removeprefix("joint.").replace("_", " ").title() + " Joint"
    )


class SimulationSceneRenderer:
    """Draw one retained simulation without owning UI or playback state."""

    def __init__(self, view: Any, chart_color: Callable[[int], str]) -> None:
        self._view = view
        self._chart_color = chart_color

    def draw(self) -> None:
        """Redraw the complete current scene while preserving the camera."""
        view = self._view
        axes = view._axes
        elev, azim = float(axes.elev), float(axes.azim)
        axes.clear()
        view._rendered_ball_center_m = None
        view._tee_artist_count = 0
        run = view._run
        if run is None:
            view._screw_readout.setVisible(False)
            axes.set_title("Run a simulation to populate the scene")
            view._canvas.draw_idle()
            return

        swing_end = float(run.swing_times[-1])
        impact_time_s = run.impact_time_s
        in_flight = impact_time_s is not None and view._time > impact_time_s
        index = int(
            np.searchsorted(run.swing_times, min(view._time, swing_end), side="left")
        )
        index = min(index, len(run.swing_times) - 1)
        show_flight = view._flight_check.isChecked() and bool(len(run.flight_positions))
        extent = self._scene_extent(in_flight, show_flight)

        if view._ground_check.isChecked():
            self._draw_ground(extent)
        tee_artists = draw_representative_tee(
            axes,
            run.config.ball_setup,
            view._display,
            self._chart_color(6),
        )
        view._tee_artist_count = len(tee_artists)
        if view._ball_check.isChecked():
            self._draw_ball(run.config.ball_position_m)
        self._draw_swing(index)
        if run.swing_joints.shape[1] >= 2:
            joints = view._display(run.swing_joints[index])
            draw_pendulum_skeleton(axes, joints, self._chart_color)
        if show_flight:
            assert impact_time_s is not None
            self._draw_flight(index, impact_time_s)

        show_screw = view._screw_check.isChecked() and not in_flight
        view._screw_readout.setVisible(show_screw)
        if show_screw:
            self._draw_screw_axis(index, extent)
        if view._kinetics_check.isChecked() and not in_flight:
            self._draw_kinetics(index)
        if view._wedge_clearance is not None and not in_flight:
            draw_wedge_ground_overlay_3d(
                axes,
                view._wedge_clearance,
                min(view._time, swing_end),
                view._display,
                self._chart_color,
            )
        if (
            view._impact_scene is not None
            and view._impact_check.isChecked()
            and not in_flight
        ):
            draw_impact_scene_3d(
                axes,
                view._impact_scene,
                view._display,
                self._chart_color,
            )
        self._finish_axes(extent, elev, azim, in_flight, show_flight)

    def _scene_extent(self, in_flight: bool, show_flight: bool) -> float:
        view = self._view
        run = view._run
        assert run is not None
        if in_flight and show_flight:
            return max(5.0, float(np.max(np.abs(run.flight_positions))) * 1.05)
        joint_extent = (
            float(np.max(np.abs(run.swing_joints))) if run.swing_joints.size else 0.0
        )
        return max(
            1.0,
            float(np.max(np.abs(run.swing_positions))) * 1.1,
            joint_extent * 1.1,
        )

    def _draw_ball(self, center_m: np.ndarray) -> None:
        view = self._view
        view._rendered_ball_center_m = np.asarray(center_m, dtype=float).copy()
        u = np.linspace(0.0, 2.0 * np.pi, 16)
        v = np.linspace(0.0, np.pi, 12)
        radius = _BALL_DRAW_RADIUS_M
        x = center_m[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center_m[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center_m[2] + radius * np.outer(np.ones_like(u), np.cos(v))
        points = view._display(np.stack([x, y, z], axis=-1))
        view._axes.plot_surface(
            points[..., 0],
            points[..., 1],
            points[..., 2],
            color=self._chart_color(4),
            alpha=0.9,
            linewidth=0.0,
            shade=True,
        )

    def _draw_ground(self, extent: float) -> None:
        view = self._view
        draw_course_ground_3d(
            view._axes,
            extent,
            layout=view._course_layout,
            elements=view._course_check.isChecked(),
        )

    def _draw_swing(self, index: int) -> None:
        view = self._view
        run = view._run
        assert run is not None
        full = view._display(run.swing_positions)
        view._axes.plot(
            full[:, 0],
            full[:, 1],
            full[:, 2],
            color=self._chart_color(0),
            lw=0.8,
            alpha=0.35,
        )
        done = view._display(run.swing_positions[: index + 1])
        view._axes.plot(
            done[:, 0],
            done[:, 1],
            done[:, 2],
            color=self._chart_color(0),
            lw=2.0,
            label="clubhead path",
        )
        head = view._display(run.swing_positions[index])
        view._axes.scatter(*head, color=self._chart_color(1), s=45, zorder=5)

    def _draw_flight(self, _index: int, impact_time_s: float) -> None:
        view = self._view
        run = view._run
        assert run is not None
        flight_t = view._time - impact_time_s
        count = int(np.searchsorted(run.flight_times, flight_t, side="right"))
        trajectory = view._display(run.flight_positions)
        view._axes.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=self._chart_color(2),
            lw=0.8,
            alpha=0.35,
        )
        if count <= 1:
            return
        done = view._display(run.flight_positions[:count])
        view._axes.plot(
            done[:, 0],
            done[:, 1],
            done[:, 2],
            color=self._chart_color(2),
            lw=2.0,
            label="ball flight",
        )
        ball = view._display(run.flight_positions[count - 1])
        view._axes.scatter(*ball, color=self._chart_color(4), s=25, zorder=5)

    def _draw_kinetics(self, index: int) -> None:
        view = self._view
        if view._run is None:
            return
        if view._kinetics is None:
            view._kinetics = kinetics_for_presentation(view._run) or False
        if not isinstance(view._kinetics, KineticsSeries):
            return
        frame = overlay_frame(view._kinetics, index)
        for item_index, (label, points) in enumerate(frame.arcs):
            display = view._display(points).T
            view._axes.plot(
                *display,
                color=self._chart_color(3 + item_index),
                lw=2.2,
                label=label,
            )
        style = {"lw": 1.6, "arrow_length_ratio": 0.18}
        for item_index, (label, start, vector) in enumerate(frame.arrows):
            start_display = view._display(start)
            vector_display = view._display(vector)
            view._axes.quiver(
                *start_display,
                *vector_display,
                color=self._chart_color(6 + item_index % 2),
                label=label,
                **style,
            )

    def _draw_screw_axis(self, index: int, extent: float) -> None:
        view = self._view
        label, motion, residual = self._selected_motion(index)
        assert view._run is not None
        view._screw_readout.setText(
            format_screw_readout(
                label, motion, view._run.config.club.loft_deg, residual
            )
        )
        ScrewOverlayRenderer(view._axes, view._display, self._chart_color).draw(
            motion, extent, label
        )

    def _joint_series(self) -> JointMotionSeries | None:
        """Lazily reconstruct joint contributions for articulated sources."""
        view = self._view
        run = view._run
        if run is None or run.swing_joints.shape[1] < 2:
            return None
        if view._joint_motion is None:
            count = run.swing_joints.shape[1] - 1
            identifiers = run.swing_joint_ids or fallback_joint_ids(count)
            view._joint_motion = analyze_joint_motion(
                run.swing_times, run.swing_joints, identifiers
            )
        return view._joint_motion

    def _selected_motion(self, index: int) -> tuple[str, ScrewMotion, float | None]:
        """Return label, motion, and optional joint reconstruction residual."""
        view = self._view
        run = view._run
        assert run is not None
        entity = str(view._screw_entity.currentData() or "club")
        if entity == "club":
            motion = analyze_twist(run.swing_twists[index], run.swing_positions[index])
            return "Club", motion, None
        series = self._joint_series()
        assert series is not None
        joint_index = series.joint_ids.index(entity)
        twist = np.concatenate(
            [
                series.angular_velocity_rad_s[index, joint_index],
                series.contribution_velocity_m_s[index, joint_index],
            ]
        )
        motion = analyze_twist(twist, run.swing_positions[index])
        return (
            joint_label(entity),
            motion,
            float(series.reconstruction_residual_m_s[index]),
        )

    def _finish_axes(
        self,
        extent: float,
        elev: float,
        azim: float,
        in_flight: bool,
        show_flight: bool,
    ) -> None:
        view = self._view
        run = view._run
        assert run is not None
        axes = view._axes
        axes.set_xlim(-extent, extent)
        axes.set_ylim(-extent, extent)
        flight_scale = in_flight and show_flight
        axes.set_zlim(0.0 if flight_scale else -extent * 0.4, extent)
        axes.set_box_aspect((2.0, 2.0, 1.0 if flight_scale else 1.4))
        axes.view_init(elev=elev, azim=azim)
        axes.set_xlabel("z — right of target [m]")
        axes.set_ylabel("x — target line [m]")
        axes.set_zlabel("y — up [m]")
        phase = "flight" if in_flight else "swing"
        if run.impact_time_s is None:
            title = (
                f"t = {view._time:.3f} s ({phase}) — no impact; "
                f"closest approach at {run.impact_outcome.candidate_time_s:.3f} s"
            )
        else:
            title = (
                f"t = {view._time:.3f} s ({phase}) — "
                f"impact at {run.impact_time_s:.3f} s"
            )
        axes.set_title(title)
        axes.legend(loc="upper left", fontsize=8)
        view._canvas.draw_idle()
