"""Animated 3D clubhead view.

The view renders procedural or STL geometry under the delivery transform,
with playback, fixed/moving display modes, velocity vectors, hosel/shaft
alignment, and an engineering-style center-of-gravity marker.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.club.volumetrics import is_watertight, mesh_volume_centroid
from rate_of_closure.mesh import HeadMesh, load_head_mesh
from rate_of_closure.model import ImpactScenario, solve
from rate_of_closure.ui.pyqt6.engineering_markers import draw_cg_marker
from rate_of_closure.ui.pyqt6.figure_canvas import (
    LifecycleSafeFigureCanvas as FigureCanvas,
)
from rate_of_closure.units import FIELD_GUIDANCE

logger = logging.getLogger(__name__)

__all__ = ["VIEW_MODES", "Club3DView"]

# Fallback palette (theme-neutral, matches shared CHART_COLORS hues).
_COL_FACE = "#0A84FF"
_COL_BODY = "#8b949e"
_COL_SHAFT = "#AC8E68"
_COL_V_REF = "#30D158"
_COL_V_POINT = "#FF375F"
_COL_IMPACT = "#FFD60A"
_COL_GROUND = "#8b949e"
_COL_COG = "#FF9F0A"

# Simplified driver-head dimensions [m].
_FACE_HALF_WIDTH = 0.058
_FACE_HALF_HEIGHT = 0.028
_BODY_DEPTH = 0.11
_SHAFT_STUB = 0.35

# STL-mesh shading: fixed world-frame light and a steel-gray base tint.
# Kept identical to the web clone (src/components/ClubCanvas.tsx).
_LIGHT_DIR = np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5])
_MESH_BASE_RGB = np.array([0.56, 0.62, 0.70])
_MESH_AMBIENT = 0.22
_MESH_SPECULAR = 0.32

_ANIMATION_SPAN_MS = 8.0
_ANIMATION_STEPS = 48
_TIMER_INTERVAL_MS = 40

#: Display modes for the 3D animation, in combo-box order.
VIEW_MODES: tuple[str, ...] = (
    "Head Fixed in Place",
    "Head Moving Through Space",
)


def _rodrigues(axis_omega: np.ndarray, dt: float) -> np.ndarray:
    """Rotation matrix for spinning at ``axis_omega`` [rad/s] for ``dt`` s."""
    theta = float(np.linalg.norm(axis_omega)) * dt
    if abs(theta) < 1e-12:
        return cast(np.ndarray, np.eye(3))
    axis = axis_omega / np.linalg.norm(axis_omega)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
    return cast(np.ndarray, rotation)


def _head_wireframe(scenario: ImpactScenario) -> dict[str, np.ndarray]:
    """Line strips describing the head at square impact, reference at origin.

    AffineDrift frame: x along the target line, y up, z right of target
    (toe side for a right-handed golfer).
    """
    d = scenario.com_to_face_mm / 1000.0
    w, h = _FACE_HALF_WIDTH, _FACE_HALF_HEIGHT
    face = np.array(
        [
            [d, -h, -w],
            [d, -h, w],
            [d, h, w],
            [d, h, -w],
            [d, -h, -w],
        ]
    )
    back = face - np.array([_BODY_DEPTH, 0.0, 0.0])
    shaft_dir = np.array(
        [
            0.0,
            np.sin(np.radians(scenario.lie_angle_deg)),
            -np.cos(np.radians(scenario.lie_angle_deg)),
        ]
    )
    hosel = np.array([d - 0.02, h, -w])
    shaft = np.vstack([hosel, hosel + shaft_dir * _SHAFT_STUB])
    impact = np.array(
        [
            d,
            scenario.impact_offset_high_mm / 1000.0,
            scenario.impact_offset_toe_mm / 1000.0,
        ]
    )
    return {"face": face, "back": back, "shaft": shaft, "impact": impact}


def _display(points: np.ndarray) -> np.ndarray:
    """Model frame (x target, y up, z right) -> matplotlib display axes.

    Matplotlib draws its z axis vertically, so plot (z, x, y): right of
    target across, target line into the page, up truly up.
    """
    return np.asarray(points)[..., [2, 0, 1]]


class Club3DView(QWidget):
    """Animated 3D rendering of the rotating clubhead at impact."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.add_subplot(111, projection="3d")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._build_playback_bar())
        layout.addWidget(self._canvas)

        self._scenario: ImpactScenario | None = None
        self._mesh: HeadMesh | None = None
        self._hosel: np.ndarray | None = None
        self._cog: np.ndarray | None = None
        self._phase = 0.0
        self._speed = 1.0
        self._zoom = 1.0
        self._timer = QTimer(self)
        self._timer.setInterval(_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)
        # Scroll-to-zoom; drag-to-orbit is native to Axes3D and the view
        # angles are captured/restored across animation redraws.
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

    # ── construction ────────────────────────────────────────────────
    def _build_playback_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 4, 4, 0)

        self._play_button = QPushButton("Play")
        self._play_button.setToolTip(
            "Play or pause the impact animation (the label shows the "
            "action the button will take)."
        )
        self._play_button.setCheckable(True)
        self._play_button.setFixedWidth(72)
        self._play_button.toggled.connect(self._on_play_toggled)
        bar.addWidget(self._play_button)

        bar.addWidget(QLabel("Playback Speed"))
        self._speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._speed_slider.setRange(10, 300)
        self._speed_slider.setValue(100)
        self._speed_slider.setToolTip("Animation speed: 0.1x to 3.0x")
        self._speed_slider.valueChanged.connect(self._on_speed_changed)
        bar.addWidget(self._speed_slider, stretch=1)
        self._speed_label = QLabel("1.0x")
        self._speed_label.setFixedWidth(40)
        bar.addWidget(self._speed_label)

        bar.addWidget(QLabel("Display"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(VIEW_MODES)
        self._mode_combo.setCurrentIndex(1)  # Head Moving is the default
        self._mode_combo.setToolTip(
            "Fixed: rotation only, easiest to read.\n"
            "Moving: the head also translates down the target line at the "
            "delivery speed."
        )
        self._mode_combo.currentTextChanged.connect(lambda _t: self._draw())
        bar.addWidget(self._mode_combo)

        self._load_mesh_button = QPushButton("Load Clubhead STL…")
        self._load_mesh_button.setToolTip(
            "Render a user-supplied STL clubhead mesh in place of the "
            "procedural wireframe (normalized to the head envelope)."
        )
        self._load_mesh_button.clicked.connect(self._on_load_mesh_clicked)
        bar.addWidget(self._load_mesh_button)

        self._reset_mesh_button = QPushButton("Procedural Head")
        self._reset_mesh_button.setToolTip("Return to the default wireframe head.")
        self._reset_mesh_button.setEnabled(False)
        self._reset_mesh_button.clicked.connect(self.clear_mesh)
        bar.addWidget(self._reset_mesh_button)

        self._show_cg_check = QCheckBox("Show CG")
        self._show_cg_check.setChecked(True)
        self._show_cg_check.setToolTip(FIELD_GUIDANCE["show_cg_marker"])
        self._show_cg_check.toggled.connect(lambda _checked: self._draw())
        bar.addWidget(self._show_cg_check)
        return bar

    # ── public API ──────────────────────────────────────────────────
    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Adopt a new scenario without starting background animation."""
        self._scenario = scenario
        self._phase = 0.0
        self._draw()

    def set_playback_speed(self, multiplier: float) -> None:
        """Set the animation speed multiplier (0.1-3.0)."""
        clamped = max(0.1, min(3.0, multiplier))
        self._speed_slider.setValue(round(clamped * 100))

    def playback_speed(self) -> float:
        """Current animation speed multiplier."""
        return self._speed

    def is_playing(self) -> bool:
        """Whether the clubhead animation timer is running."""
        return self._timer.isActive()

    def set_view_mode(self, mode: str) -> None:
        """Select a display mode by name (see :data:`VIEW_MODES`)."""
        if mode not in VIEW_MODES:
            logger.warning("unknown view mode requested: %s", mode)
            return
        self._mode_combo.setCurrentText(mode)

    def view_mode(self) -> str:
        """The active display mode name."""
        return self._mode_combo.currentText()

    def set_zoom(self, factor: float) -> None:
        """Set the camera zoom factor (0.3-4.0; larger = closer)."""
        self._zoom = max(0.3, min(4.0, factor))
        self._draw()

    def zoom(self) -> float:
        """Current camera zoom factor."""
        return self._zoom

    def load_mesh(self, path: str) -> None:
        """Load an STL clubhead mesh and switch to photorealistic mode.

        Raises the mesh module's contract errors on unparseable files
        (the button handler wraps them in a dialog). A watertight STL
        gets its volumetric COG; otherwise the CG marker falls back to
        the reference point (the spec CG)."""
        mesh = load_head_mesh(path)
        cog: tuple[float, float, float] | None = None
        if is_watertight(mesh.triangles):
            _volume, centroid = mesh_volume_centroid(mesh.triangles)
            cog = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
        self.set_head_mesh(mesh, cog_point=cog)

    def set_head_mesh(
        self,
        mesh: HeadMesh,
        hosel_point: tuple[float, float, float] | None = None,
        cog_point: tuple[float, float, float] | None = None,
    ) -> None:
        """Render a prepared head mesh (STL or parametric) in the view.

        The Club group's "Generate Representative Head" passes its
        per-type hosel point (where the shaft line attaches) and its
        volumetric COG; STL loads leave them ``None``.
        """
        self._mesh = mesh
        self._hosel = None if hosel_point is None else np.asarray(hosel_point)
        self._cog = None if cog_point is None else np.asarray(cog_point)
        self._reset_mesh_button.setEnabled(True)
        self._draw()

    def clear_mesh(self) -> None:
        """Discard any loaded STL mesh and restore the procedural head."""
        self._mesh = None
        self._hosel = None
        self._cog = None
        self._reset_mesh_button.setEnabled(False)
        self._draw()

    def show_cg_check(self) -> QCheckBox:
        """The 'Show CG' checkbox (test seam)."""
        return self._show_cg_check

    def shaft_attachment(self) -> np.ndarray | None:
        """Model-frame shaft attachment: the generated head's hosel
        shifted with the mesh; ``None`` for the wireframe hosel."""
        if self._scenario is None or self._mesh is None or self._hosel is None:
            return None
        return np.asarray(self._hosel + self._head_shift(self._mesh, self._scenario))

    def cg_marker_point(self) -> np.ndarray | None:
        """Model-frame CG marker location, or ``None`` when hidden."""
        if self._scenario is None or not self._show_cg_check.isChecked():
            return None
        if self._mesh is None or self._cog is None:
            return np.zeros(3)  # spec CG fallback: the reference point
        return np.asarray(self._cog + self._head_shift(self._mesh, self._scenario))

    @staticmethod
    def _head_shift(mesh: HeadMesh, scenario: ImpactScenario) -> np.ndarray:
        """+x shift placing the mesh's face plane at GC-to-face."""
        d = scenario.com_to_face_mm / 1000.0
        return np.array([d - float(mesh.triangles[..., 0].max()), 0.0, 0.0])

    def has_mesh(self) -> bool:
        """Whether an STL mesh is currently rendered."""
        return self._mesh is not None

    def stop(self) -> None:
        """Stop the animation timer (used on window close and in tests)."""
        self._timer.stop()
        self._play_button.setChecked(False)

    # ── internals ──────────────────────────────────────────────────
    def _on_load_mesh_clicked(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Load Clubhead STL", "", "STL meshes (*.stl);;All files (*)"
        )
        if not path:
            return
        try:
            self.load_mesh(path)
        except Exception as exc:  # noqa: BLE001 — surface any parse failure
            logger.warning("STL load failed: %s", exc)
            QMessageBox.warning(self, "STL Load Failed", str(exc))

    def _on_scroll(self, event) -> None:  # type: ignore[no-untyped-def]
        self.set_zoom(self._zoom * (1.1 if event.button == "up" else 1.0 / 1.1))

    def _on_play_toggled(self, playing: bool) -> None:
        self._play_button.setText("Pause" if playing else "Play")
        if playing and self._scenario is not None:
            self._timer.start()
        else:
            self._timer.stop()

    def _on_speed_changed(self, value: int) -> None:
        self._speed = value / 100.0
        self._speed_label.setText(f"{self._speed:.1f}x")

    def _advance(self) -> None:
        self._phase = (self._phase + self._speed / _ANIMATION_STEPS) % 1.0
        self._draw()

    def _draw(self) -> None:
        if self._scenario is None:
            return
        scenario = self._scenario
        result = solve(scenario)
        omega = np.radians(np.array(result.omega_dps))
        time_s = (self._phase - 0.5) * _ANIMATION_SPAN_MS / 1000.0
        rotation = _rodrigues(omega, time_s)
        moving = self._mode_combo.currentText() == VIEW_MODES[1]
        speed_mps = result.reference_speed_mph * 0.44704
        offset = np.array([speed_mps * time_s, 0.0, 0.0]) if moving else np.zeros(3)

        parts = _head_wireframe(scenario)
        axes = self._axes
        # Preserve the user's orbit angles across the animation redraw.
        elev, azim = float(axes.elev), float(axes.azim)
        axes.clear()
        if self._mesh is not None:
            self._draw_mesh(self._mesh, scenario, rotation, offset)
            shaft = parts["shaft"]
            attachment = self.shaft_attachment()
            if attachment is not None:
                # Hosel-true shaft: attach at the per-type hosel point.
                lie = np.radians(scenario.lie_angle_deg)
                shaft_dir = np.array([0.0, np.sin(lie), -np.cos(lie)])
                shaft = np.vstack([attachment, attachment + shaft_dir * _SHAFT_STUB])
            pts = _display(shaft @ rotation.T + offset)
            axes.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=_COL_SHAFT, lw=2.0)
        else:
            for key, color, width in (
                ("face", _COL_FACE, 2.2),
                ("back", _COL_BODY, 1.2),
                ("shaft", _COL_SHAFT, 2.0),
            ):
                pts = _display(parts[key] @ rotation.T + offset)
                axes.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=width)
            for a, b in zip(parts["face"], parts["back"], strict=True):
                seg = _display(np.vstack([a, b]) @ rotation.T + offset)
                axes.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=_COL_BODY, lw=0.8)

        impact = parts["impact"] @ rotation.T + offset
        axes.scatter(*_display(impact), color=_COL_IMPACT, s=45, zorder=5)
        axes.scatter(*_display(offset), color=_COL_BODY, s=30)
        cg_point = self.cg_marker_point()
        if cg_point is not None:
            placed = _display(cg_point @ rotation.T + offset)
            draw_cg_marker(axes, placed, _COL_COG)

        if moving:
            # Target line on the ground plane, for spatial reference.
            line = _display(np.array([[-0.4, -0.05, 0.0], [0.4, -0.05, 0.0]]))
            axes.plot(
                line[:, 0],
                line[:, 1],
                line[:, 2],
                color=_COL_GROUND,
                lw=0.8,
                ls=":",
            )

        scale = 0.0035  # m per (m/s): keeps arrows inside the box
        v_ref = np.array([result.reference_speed_mph, 0.0, 0.0]) * 0.44704
        v_point = np.array(result.point_velocity_mps)
        for origin, vec, color, label in (
            (offset, v_ref, _COL_V_REF, "reference (GC) path"),
            (impact, v_point, _COL_V_POINT, "impact-point path"),
        ):
            axes.quiver(
                *_display(origin),
                *_display(vec * scale),
                color=color,
                lw=2.0,
                arrow_length_ratio=0.22,
                label=label,
            )

        limit = (0.24 if not moving else 0.42) / self._zoom
        axes.set_xlim(-limit, limit)
        axes.set_ylim(-limit * 0.6, limit * 1.4)
        axes.set_zlim(-limit * 0.6, limit * 1.4)
        axes.view_init(elev=elev, azim=azim)
        axes.set_xlabel("z — right of target [m]")
        axes.set_ylabel("x — target line [m]")
        axes.set_zlabel("y — up [m]")
        axes.set_title(
            f"Path Δ {result.path_deviation_deg:+.2f}°   "
            f"AoA Δ {result.aoa_deviation_deg:+.2f}°   "
            f"t = {time_s * 1000.0:+.1f} ms"
        )
        axes.legend(loc="upper left", fontsize=8)
        self._canvas.draw_idle()

    def _draw_mesh(
        self,
        mesh: HeadMesh,
        scenario: ImpactScenario,
        rotation: np.ndarray,
        offset: np.ndarray,
    ) -> None:
        """Shaded STL head under the same transform as the wireframe.

        The mesh is shifted by :meth:`_head_shift` so its face plane
        sits at ``com_to_face``, then rotated about the reference point
        and translated with the head. Shading is flat lambert-ish:
        ``ambient + (1 - ambient) * |n . L|`` with a fixed world light
        on the rotated normals; depth ordering is Poly3DCollection's
        native painter's-algorithm z-sort.
        """
        tris = (mesh.triangles + self._head_shift(mesh, scenario)) @ rotation.T + offset
        normals = mesh.normals @ rotation.T
        lambert = np.abs(normals @ _LIGHT_DIR)
        diffuse = (1.0 - _MESH_AMBIENT - _MESH_SPECULAR) * lambert
        specular = _MESH_SPECULAR * lambert**20
        intensity = _MESH_AMBIENT + diffuse + specular
        colors = np.clip(intensity[:, None] * _MESH_BASE_RGB[None, :], 0.0, 1.0)
        collection = Poly3DCollection(
            _display(tris), facecolors=colors, edgecolors="none", linewidths=0.0
        )
        self._axes.add_collection3d(collection)
