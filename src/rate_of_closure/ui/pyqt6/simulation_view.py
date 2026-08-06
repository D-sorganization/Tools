"""Swing-scale 3D scene and playback controls.

Renders the swing skeleton, path, ball, ground, screw axis, and flight overlays.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
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
    ground_clearance_snapshot_for_scene,
)
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)
from rate_of_closure.ui.pyqt6.simulation_scene_renderer import (
    SimulationSceneRenderer,
    fallback_joint_ids,
    joint_label,
)
from rate_of_closure.ui.pyqt6.simulation_specs import RATE_PRESETS
from rate_of_closure.units import FIELD_GUIDANCE

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
_IMPACT_LAYER_SETTINGS_ORG = "RateOfClosure"
_IMPACT_LAYER_SETTINGS_APP = "ImpactScene"
_IMPACT_LAYER_SETTINGS_KEY = "visible_layers_v1"
_DEFAULT_IMPACT_LAYERS = frozenset(
    {"face_normal", "face_center_travel", "dplane_normal", "spin_loft_sector"}
)


class SimulationView(QWidget):
    """Animated 3D scene of one simulation run with video controls."""

    def __init__(
        self,
        parent: QWidget | None = None,
        impact_settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self._impact_settings = impact_settings or QSettings(
            _IMPACT_LAYER_SETTINGS_ORG, _IMPACT_LAYER_SETTINGS_APP
        )
        self._figure = Figure(figsize=(5, 5), tight_layout=True)
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
        layout.addLayout(self._build_playback_bar())
        layout.addLayout(self._build_toggle_bar())
        self._screw_readout = QLabel()
        self._screw_readout.setWordWrap(True)
        self._screw_readout.setVisible(False)
        self._screw_readout.setToolTip(
            "Engineering screw-motion readout in the app frame: x target, "
            "y up, z right. Orbital + axial velocity reconstructs the selected "
            "point velocity. Pure translation has an axis at infinity."
        )
        layout.addWidget(self._screw_readout)
        self._impact_kinematics_readout = QLabel()
        self._impact_kinematics_readout.setWordWrap(True)
        self._impact_kinematics_readout.setTextFormat(Qt.TextFormat.RichText)
        self._impact_kinematics_readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._impact_kinematics_readout.setMargin(8)
        self._impact_kinematics_readout.setStyleSheet(
            "QLabel { background: palette(base); border: 1px solid palette(mid); "
            "border-radius: 5px; }"
        )
        self._impact_kinematics_readout.setAccessibleName(
            "Impact and Wedge Engineering Readout"
        )
        self._impact_kinematics_readout.setToolTip(
            "Frame-explicit contact-point, shaft-rotation, face-normal, leading-edge, "
            "screw-axis, and wedge ground-clearance metrics. Misses are evaluated "
            "only at closest approach; ground clearance remains geometry-only."
        )
        layout.addWidget(self._impact_kinematics_readout)
        layout.addWidget(self._canvas)

        self._timer = QTimer(self)
        self._timer.setInterval(_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)

    def _build_playback_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 4, 4, 0)

        self._play_button = QPushButton("Play")
        self._play_button.setCheckable(True)
        self._play_button.setFixedWidth(64)
        self._play_button.setToolTip("Play or pause the swing + flight playback.")
        self._play_button.toggled.connect(self._on_play_toggled)
        bar.addWidget(self._play_button)

        self._step_back_button = QPushButton("−1 frame")
        self._step_back_button.setToolTip("Step one sample backward.")
        self._step_back_button.clicked.connect(lambda: self.step_frames(-1))
        bar.addWidget(self._step_back_button)

        self._step_forward_button = QPushButton("+1 frame")
        self._step_forward_button.setToolTip("Step one sample forward.")
        self._step_forward_button.clicked.connect(lambda: self.step_frames(1))
        bar.addWidget(self._step_forward_button)

        self._inspection_button = QPushButton("Jump to Impact")
        self._inspection_button.setEnabled(False)
        self._inspection_button.setToolTip(
            "Jump to the exact impact event; for a miss, jump to the explicitly "
            "labeled sampled closest approach."
        )
        self._inspection_button.clicked.connect(self.jump_to_inspection_event)
        bar.addWidget(self._inspection_button)

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setRange(0, _SLIDER_STEPS)
        self._position_slider.setToolTip(
            "Scrub the playback instant across the whole swing + flight timeline."
        )
        self._position_slider.valueChanged.connect(self._on_slider_moved)
        bar.addWidget(self._position_slider, stretch=1)

        self._time_label = QLabel("0.000 s")
        self._time_label.setFixedWidth(72)
        bar.addWidget(self._time_label)

        self._loop_check = QCheckBox("Loop")
        self._loop_check.setToolTip("Restart playback when the timeline ends.")
        bar.addWidget(self._loop_check)

        bar.addWidget(QLabel("Rate"))
        self._rate_combo = QComboBox()
        self._rate_combo.addItems([name for name, _ in RATE_PRESETS])
        self._rate_combo.setCurrentIndex(3)  # 1x real-time
        self._rate_combo.setToolTip(
            "Playback rate: 1× maps animation time to simulated time; "
            "slower presets reveal the impact window."
        )
        bar.addWidget(self._rate_combo)
        return bar

    def _build_toggle_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 0)
        self._ball_check = QCheckBox("Ball")
        self._ball_check.setChecked(True)
        self._ball_check.setToolTip(FIELD_GUIDANCE["ball_visible"])
        self._ground_check = QCheckBox("Ground")
        self._ground_check.setChecked(True)
        self._ground_check.setToolTip(FIELD_GUIDANCE["ground_visible"])
        # Course scene (#4125 H7a): fairway strip, green + flag, tee.
        self._course_check = QCheckBox("Course Elements")
        self._course_check.setChecked(True)
        self._course_check.setToolTip(FIELD_GUIDANCE["course_visible"])
        self._screw_check = QCheckBox("Screw Axis")
        self._screw_check.setChecked(False)
        self._screw_check.setToolTip(FIELD_GUIDANCE["screw_axis_visible"])
        self._impact_check = QCheckBox("Impact Inspector")
        self._impact_check.setChecked(True)
        self._impact_check.setToolTip(
            "Show the exact event-time wedge face, physical shaft axis, contact point, "
            "face normal, leading edge, arc tangent, screw axis, and velocity "
            "decomposition."
        )
        saved_layers = self._load_impact_layers()
        self._impact_layer_checks: dict[str, QCheckBox] = {}
        for key, label, tooltip in (
            (
                "face_normal",
                "Face Normal",
                "Show the delivered face-center normal in the app frame.",
            ),
            (
                "face_center_travel",
                "Face-Center Travel",
                "Show rigid-body face-center travel including omega cross r.",
            ),
            (
                "dplane_normal",
                "D-Plane Normal",
                "Show the normal to the plane spanned by face-center travel "
                "and normal.",
            ),
            (
                "spin_loft_sector",
                "Spin Loft",
                "Show the shaded exact 3D angle between face-center travel and normal.",
            ),
        ):
            check = QCheckBox(label)
            check.setChecked(key in saved_layers)
            check.setToolTip(
                "Suggested range: on for engineering review; turn off to isolate "
                "other layers. "
                + tooltip
                + " Source: standard 3D D-plane geometry. Frame: x target, "
                "y up, z right."
            )
            check.toggled.connect(self._on_impact_layer_toggled)
            self._impact_layer_checks[key] = check
        self._impact_view = QComboBox()
        self._impact_view.setAccessibleName("Impact Camera View")
        self._impact_view.addItem("Isometric", (30.0, -60.0))
        self._impact_view.addItem("Face-On", (0.0, -90.0))
        self._impact_view.addItem("Down-the-Line", (10.0, 0.0))
        self._impact_view.setToolTip(
            "Select a named engineering camera; the plot remains freely rotatable."
        )
        self._impact_view.currentIndexChanged.connect(self._apply_impact_view)
        self._impact_export_button = QPushButton("Export Impact…")
        self._impact_export_button.setEnabled(False)
        self._impact_export_button.setToolTip(
            "Export the exact impact still as high-resolution PNG, vector SVG, "
            "or the versioned scene data as JSON."
        )
        self._impact_export_button.clicked.connect(self._on_export_impact)
        self._screw_entity = QComboBox()
        self._screw_entity.addItem("Club", "club")
        self._screw_entity.setToolTip(
            "Select the club rigid body or a modeled revolute joint. "
            "Joint glyphs show that joint's instantaneous contribution at the clubhead."
        )
        self._screw_entity.currentIndexChanged.connect(lambda _index: self._draw())
        # Kinetics overlay (#4125 H2): torque arcs + force arrows.
        self._kinetics_check = QCheckBox("Show Kinetics")
        self._kinetics_check.setChecked(False)
        self._kinetics_check.setToolTip(FIELD_GUIDANCE["kinetics_visible"])
        # Scale separation (epic #4120): flight display is opt-in here
        # because its envelope dwarfs the swing envelope.
        self._flight_check = QCheckBox("Show Ball Flight")
        self._flight_check.setChecked(False)
        self._flight_check.setToolTip(
            "Warning: turning this on expands the scene to flight scale "
            "(100+ m), which dwarfs the ~3 m swing. "
            + FIELD_GUIDANCE["swing_flight_toggle"]
        )
        for check in (
            self._ball_check,
            self._ground_check,
            self._course_check,
            self._screw_check,
            self._impact_check,
            self._kinetics_check,
            self._flight_check,
        ):
            check.toggled.connect(lambda _checked: self._draw())
            bar.addWidget(check)
        for check in self._impact_layer_checks.values():
            bar.addWidget(check)
        bar.addWidget(self._screw_entity)
        bar.addWidget(self._impact_view)
        bar.addWidget(self._impact_export_button)
        bar.addStretch(1)
        return bar

    def _load_impact_layers(self) -> frozenset[str]:
        raw = self._impact_settings.value(_IMPACT_LAYER_SETTINGS_KEY)
        if not isinstance(raw, str):
            return _DEFAULT_IMPACT_LAYERS
        try:
            values = json.loads(raw)
        except (TypeError, ValueError):
            return _DEFAULT_IMPACT_LAYERS
        allowed = _DEFAULT_IMPACT_LAYERS
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value in allowed for value in values
        ):
            return _DEFAULT_IMPACT_LAYERS
        return frozenset(values)

    def _on_impact_layer_toggled(self, _checked: bool) -> None:
        self._impact_settings.setValue(
            _IMPACT_LAYER_SETTINGS_KEY,
            json.dumps(sorted(self.impact_visible_layers())),
        )
        self._draw()

    def impact_visible_layers(self) -> frozenset[str]:
        """Return persisted, independently toggleable impact layers."""
        return frozenset(
            key for key, check in self._impact_layer_checks.items() if check.isChecked()
        )

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
        index = int(self._rate_combo.currentIndex())
        return float(RATE_PRESETS[index][1])

    def set_playback_rate(self, multiplier: float) -> None:
        """Select the nearest rate preset to ``multiplier``."""
        index = int(np.argmin([abs(rate - multiplier) for _, rate in RATE_PRESETS]))
        self._rate_combo.setCurrentIndex(index)

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

    def is_playing(self) -> bool:
        """Whether the playback timer is running."""
        return bool(self._timer.isActive())

    def set_looping(self, looping: bool) -> None:
        """Set the loop toggle."""
        self._loop_check.setChecked(looping)

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
        elevation, azimuth = self._impact_view.itemData(index)
        self._axes.view_init(elev=float(elevation), azim=float(azimuth))
        self._draw()

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
        SimulationSceneRenderer(self, get_chart_color).draw()
