"""Swing-scale 3D scene and playback controls.

Renders the swing skeleton, path, ball, ground, screw axis, and flight overlays.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import QSettings, QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.application.camera_commands import (
    CameraCommandId,
    camera_preset,
    matplotlib_angles,
)
from rate_of_closure.simulation import (
    ImpactScene,
    KineticsSeries,
    RunGroundClearanceSnapshot,
    SimulationRun,
    impact_scene_for_run,
)
from rate_of_closure.simulation.screw_analysis import (
    JointMotionSeries,
)
from rate_of_closure.ui.course import CourseLayout
from rate_of_closure.ui.impact_kinematics_presentation import (
    format_simulation_engineering_readout,
    format_simulation_key_metrics,
    ground_clearance_snapshot_for_scene,
)
from rate_of_closure.ui.pyqt6.camera_controls import CameraViewportMixin
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)
from rate_of_closure.ui.pyqt6.impact_layer_controls import ImpactLayerControls
from rate_of_closure.ui.pyqt6.simulation_legend_layout import reflow_simulation_legend
from rate_of_closure.ui.pyqt6.simulation_scene_renderer import (
    SimulationSceneRenderer,
    fallback_joint_ids,
    joint_label,
)
from rate_of_closure.ui.pyqt6.simulation_specs import RATE_PRESETS
from rate_of_closure.ui.pyqt6.simulation_view_controls import (
    SimulationViewControlsMixin,
)

try:  # Theme palette (optional in standalone/vendored use).
    from shared.python.theme.matplotlib_style import get_chart_color
except ImportError:  # pragma: no cover - theme package always ships in-repo

    def get_chart_color(index: int) -> str:
        """Matplotlib cycle colors as a theme-neutral fallback."""
        return f"C{index % 10}"


logger = logging.getLogger(__name__)
__all__ = ["RATE_PRESETS", "SimulationView"]

_TIMER_INTERVAL_MS = 40
_SLIDER_STEPS = 1000


class SimulationView(CameraViewportMixin, SimulationViewControlsMixin, QWidget):
    """Animated 3D scene of one simulation run with video controls."""

    def __init__(
        self,
        parent: QWidget | None = None,
        impact_settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._impact_layer_controls: ImpactLayerControls = ImpactLayerControls(
            impact_settings,
            self._draw,
        )
        # Kept as a compatibility seam for existing UI automation.
        self._impact_layer_checks = self._impact_layer_controls.checks
        self._figure = Figure(figsize=(5, 5))
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")

        self._course_layout = CourseLayout()
        self._run: SimulationRun | None = None
        self._joint_motion: JointMotionSeries | None = None
        self._kinetics: KineticsSeries | None | bool = None
        self._wedge_clearance: RunGroundClearanceSnapshot | None = None
        self._impact_scene: ImpactScene | None = None
        self._time = 0.0
        self._rendered_ball_center_m: np.ndarray | None = None
        self._tee_artist_count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(self._build_playback_controls())
        layout.addWidget(self._build_layers_control())
        layout.addWidget(self._initialize_camera("Clubhead"))
        layout.addWidget(self._build_engineering_panel())
        self._canvas.setMinimumSize(360, 280)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self._canvas, stretch=1)

        self._timer = QTimer(self)
        self._timer.setInterval(_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)
        self._canvas.mpl_connect(
            "button_release_event", lambda _event: self.suspend_camera_tracking()
        )
        self._canvas.mpl_connect("resize_event", self._redraw_after_canvas_resize)

    # ── public API ──────────────────────────────────────────────────
    def set_run(self, run: SimulationRun | None) -> None:
        """Adopt a run (or clear with ``None``) and reset the timeline."""
        self._run = run
        self._joint_motion = None
        self._kinetics = None
        self._wedge_clearance = ground_clearance_snapshot_for_scene(run)
        self._impact_scene = impact_scene_for_run(run) if run is not None else None
        self._time = 0.0
        self._inspection_button.setEnabled(run is not None)
        self._impact_export_button.setEnabled(run is not None)
        self._inspection_button.setText(
            f"Jump to {run.inspection_event_label}"
            if run is not None
            else "Jump to Impact"
        )
        self._impact_kinematics_readout.setText(
            format_simulation_engineering_readout(run)
            if run is not None
            else "Run a simulation to inspect impact kinematics."
        )
        self._impact_summary.setText(format_simulation_key_metrics(run))
        self._populate_screw_entities()
        self._sync_slider()
        self._draw()

    def run(self) -> SimulationRun | None:
        """The run currently rendered, if any."""
        return self._run

    def playback_time(self) -> float:
        """Current playback instant [s] on the swing + flight timeline."""
        return self._time

    def set_playback_time(self, t: float) -> None:
        """Move the playback instant (clamped to the timeline)."""
        if self._run is None:
            return
        self._time = min(max(t, 0.0), self._run.total_duration_s)
        self._sync_slider()
        self._draw()

    def playback_rate(self) -> float:
        """The selected playback-rate multiplier."""
        return float(self._rate_spin.value())

    def set_playback_rate(self, multiplier: float) -> None:
        """Set a granular playback multiplier, clamped to the supported range."""
        self._rate_spin.setValue(multiplier)

    def step_frames(self, frames: int) -> None:
        """Step the playback instant by whole swing-sample intervals."""
        if self._run is None:
            return
        dt = float(self._run.swing_times[1] - self._run.swing_times[0])
        self.set_playback_time(self._time + frames * dt)

    def jump_to_inspection_event(self) -> None:
        """Pause and move to physical impact or a miss's closest approach."""
        if self._run is None:
            return
        self._play_button.setChecked(False)
        self.set_playback_time(self._run.inspection_time_s)

    def restart_playback(self) -> None:
        """Pause and return to the first frame."""
        self._play_button.setChecked(False)
        self.set_playback_time(0.0)

    def is_playing(self) -> bool:
        """Whether the playback timer is running."""
        return bool(self._timer.isActive())

    def set_looping(self, looping: bool) -> None:
        """Set the loop toggle."""
        self._loop_check.setChecked(looping)

    def is_looping(self) -> bool:
        """Whether playback restarts automatically at the timeline end."""
        return bool(self._loop_check.isChecked())

    def flight_shown(self) -> bool:
        """Whether the flight-scale 'Show Ball Flight' toggle is on."""
        return bool(self._flight_check.isChecked())

    def set_flight_shown(self, shown: bool) -> None:
        """Set the 'Show Ball Flight' toggle (default off: swing scale)."""
        self._flight_check.setChecked(shown)

    def course_layout(self) -> CourseLayout:
        """The course furniture layout rendered by the ground painter."""
        return self._course_layout

    def set_course_layout(self, layout: CourseLayout) -> None:
        """Adopt a course layout (H7b target edits drive this) and redraw."""
        self._course_layout = layout
        self._draw()

    def course_elements_shown(self) -> bool:
        """Whether the 'Course Elements' toggle is on."""
        return bool(self._course_check.isChecked())

    def scene_extent_m(self) -> float:
        """Current axis half-extent [m] — the scale-invariant seam."""
        x0, x1 = self._axes.get_xlim()
        return abs(float(x1) - float(x0)) / 2.0

    def rendered_ball_center_m(self) -> np.ndarray:
        """Return the canonical ball center used by the latest scene draw."""
        if self._rendered_ball_center_m is None:
            return np.zeros(3, dtype=float)
        return self._rendered_ball_center_m.copy()

    def tee_visible(self) -> bool:
        """Return whether the latest scene draw contains tee geometry."""
        return self._tee_artist_count > 0

    def export_impact_scene(self, file_path: str | Path) -> Path:
        """Export the exact-event engineering still or versioned scene data."""
        if self._run is None or self._impact_scene is None:
            raise RuntimeError("an impact scene requires a completed simulation")
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = self._impact_scene.to_json_dict()
            payload["render_preferences"] = {
                "visible_layers": sorted(self.impact_visible_layers()),
                "camera": {
                    "elevation_deg": float(self._axes.elev),
                    "azimuth_deg": float(self._axes.azim),
                },
            }
            path.write_text(
                json.dumps(payload, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            return path
        if suffix not in {".png", ".svg"}:
            raise ValueError("impact scene export must use .png, .svg, or .json")
        self.jump_to_inspection_event()
        self._figure.savefig(
            path,
            dpi=300 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor=self._figure.get_facecolor(),
        )
        return path

    def stop(self) -> None:
        """Stop the playback timer (window close and tests)."""
        self._timer.stop()
        self._play_button.setChecked(False)

    # ── internals ──────────────────────────────────────────────────
    def _on_play_toggled(self, playing: bool) -> None:
        self._play_button.setText("Pause" if playing else "Play")
        if playing and self._run is not None:
            if self._time >= self._run.total_duration_s - np.finfo(float).eps:
                self._time = 0.0
                self._sync_slider()
                self._draw()
            self._timer.start()
        else:
            self._timer.stop()

    def _on_slider_moved(self, value: int) -> None:
        if self._run is None:
            return
        t = value / _SLIDER_STEPS * self._run.total_duration_s
        if abs(t - self._time) > 1e-12:
            self._time = t
            self._draw()

    def _sync_slider(self) -> None:
        total = self._run.total_duration_s if self._run is not None else 0.0
        value = round(self._time / total * _SLIDER_STEPS) if total > 0.0 else 0
        self._position_slider.blockSignals(True)
        self._position_slider.setValue(value)
        self._position_slider.blockSignals(False)
        self._time_label.setText(f"{self._time:.3f} s")

    def _advance(self) -> None:
        if self._run is None:
            return
        self._time += _TIMER_INTERVAL_MS / 1000.0 * self.playback_rate()
        total = self._run.total_duration_s
        if self._time > total:
            if self._loop_check.isChecked():
                self._time = 0.0
            else:
                self._time = total
                self._play_button.setChecked(False)
        self._sync_slider()
        self._draw()

    def _populate_screw_entities(self) -> None:
        """Expose the club plus every articulated revolute joint in the run."""
        selected = self._screw_entity.currentData()
        self._screw_entity.blockSignals(True)
        self._screw_entity.clear()
        self._screw_entity.addItem("Club", "club")
        if self._run is not None and self._run.swing_joints.shape[1] >= 2:
            count = self._run.swing_joints.shape[1] - 1
            identifiers = self._run.swing_joint_ids or fallback_joint_ids(count)
            for joint_id in identifiers:
                self._screw_entity.addItem(joint_label(joint_id), joint_id)
        match = self._screw_entity.findData(selected)
        self._screw_entity.setCurrentIndex(max(match, 0))
        self._screw_entity.blockSignals(False)

    def _apply_impact_view(self, index: int) -> None:
        """Apply a named camera without preventing subsequent free orbit."""
        self.apply_camera_command(
            CameraCommandId(str(self._impact_view.itemData(index)))
        )

    def _camera_subject_m(self) -> tuple[float, float, float]:
        """Return the current clubhead in the canonical app frame."""
        if self._run is None or not len(self._run.swing_positions):
            return (0.0, 0.0, 0.0)
        index = int(np.searchsorted(self._run.swing_times, self._time, side="left"))
        index = min(index, len(self._run.swing_positions) - 1)
        position = self._run.swing_positions[index]
        return (float(position[0]), float(position[1]), float(position[2]))

    def _camera_base_half_extent_m(self) -> float:
        """Return the unzoomed swing envelope half extent."""
        if self._run is None:
            return 1.0
        joint_extent = (
            float(np.max(np.abs(self._run.swing_joints)))
            if self._run.swing_joints.size
            else 0.0
        )
        return max(
            1.0,
            float(np.max(np.abs(self._run.swing_positions))) * 1.1,
            joint_extent * 1.1,
        )

    @staticmethod
    def _camera_subject_radius_m() -> float:
        """Clearance envelope for the representative head and shaft stub."""
        return 0.35

    def _camera_state_changed(self) -> None:
        """Apply a camera state change without sharing it with another viewport."""
        self._draw()

    def _camera_orientation(self) -> tuple[float, float] | None:
        """Return an exact Matplotlib preset, or preserve a manual orbit."""
        command = self._camera_state.preset_id
        if command is None:
            return None
        elevation, azimuth = matplotlib_angles(
            camera_preset(command, self._camera_state.face_on_side)
        )
        return float(elevation), float(azimuth)

    def _on_export_impact(self) -> None:
        """Choose and export an impact artifact from the desktop UI."""
        selected, _filter = QFileDialog.getSaveFileName(
            self,
            "Export Impact Engineering Scene",
            "wedge-impact.svg",
            "Vector SVG (*.svg);;High-Resolution PNG (*.png);;Scene Data (*.json)",
        )
        if selected:
            self.export_impact_scene(selected)

    @staticmethod
    def _display(points: np.ndarray) -> np.ndarray:
        """App frame (x target, y up, z right) -> matplotlib display axes."""
        display: np.ndarray = np.asarray(points)[..., [2, 0, 1]]
        return display

    def _draw(self) -> None:
        self._advance_camera_tracking()
        orientation = self._camera_orientation()
        if orientation is not None:
            elevation, azimuth = orientation
            self._axes.view_init(elev=elevation, azim=azimuth)
        SimulationSceneRenderer(self, get_chart_color).draw()

    def _redraw_after_canvas_resize(self, _event: object) -> None:
        """Reflow the legend without advancing camera or playback state."""
        reflow_simulation_legend(self)
