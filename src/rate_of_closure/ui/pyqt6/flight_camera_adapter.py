"""Camera behavior isolated from the ball-flight drawing orchestrator."""

from __future__ import annotations

from mpl_toolkits.mplot3d.axes3d import Axes3D

from rate_of_closure.application.camera_commands import camera_preset, matplotlib_angles
from rate_of_closure.simulation.flight_playback import TimedTrajectory
from rate_of_closure.ui.pyqt6.camera_controls import CameraViewportMixin


class FlightCameraAdapter(CameraViewportMixin):
    """Apply the shared camera contract to one Matplotlib flight viewport."""

    _axes_3d: Axes3D | None
    _manual_orientation: tuple[float, float]
    _playback_time_s: float
    _timed_trajectory: TimedTrajectory | None

    def extents_m(self) -> tuple[float, float, float]:
        raise NotImplementedError

    def _draw(self) -> None:
        raise NotImplementedError

    def _camera_subject_m(self) -> tuple[float, float, float]:
        if self._timed_trajectory is None:
            return (0.0, 0.0, 0.0)
        position = self._timed_trajectory.frame_at(self._playback_time_s).position_m
        return (float(position[0]), float(position[1]), float(position[2]))

    def _camera_base_half_extent_m(self) -> float:
        carry, height, lateral = self.extents_m()
        return max(carry / 2.0, height / 2.0, lateral)

    @staticmethod
    def _camera_subject_radius_m() -> float:
        return 0.05

    def _camera_state_changed(self) -> None:
        self._draw()

    def _manual_camera_released(self, _event: object) -> None:
        if self._axes_3d is not None:
            self._manual_orientation = (
                float(self._axes_3d.elev),
                float(self._axes_3d.azim),
            )
        self.suspend_camera_tracking()

    def _apply_camera_to_axes(self) -> None:
        axes = self._axes_3d
        if axes is None:
            return
        command = self._camera_state.preset_id
        orientation = (
            self._manual_orientation
            if command is None
            else matplotlib_angles(
                camera_preset(command, self._camera_state.face_on_side)
            )
        )
        axes.view_init(elev=orientation[0], azim=orientation[1])
        self._apply_camera_axis_visibility(axes)
        if not self._camera_state.tracking_enabled:
            return
        carry, height, lateral = self.extents_m()
        downrange, up, right = self._camera_state.target_m
        zoom = self._camera_state.zoom
        axes.set_xlim(right - lateral / zoom, right + lateral / zoom)
        axes.set_ylim(downrange - carry / (2 * zoom), downrange + carry / (2 * zoom))
        axes.set_zlim(up - height / (2 * zoom), up + height / (2 * zoom))
        axes.set_box_aspect((2 * lateral, carry, height))

    def camera_subject_in_frame(self) -> bool:
        """Return whether the current ball lies strictly inside the 3D limits."""
        if self._axes_3d is None:
            return False
        downrange, up, right = self._camera_subject_m()
        limits = (
            self._axes_3d.get_xlim3d(),
            self._axes_3d.get_ylim3d(),
            self._axes_3d.get_zlim3d(),
        )
        return all(
            low < value < high
            for value, (low, high) in zip((right, downrange, up), limits, strict=True)
        )


__all__ = ["FlightCameraAdapter"]
