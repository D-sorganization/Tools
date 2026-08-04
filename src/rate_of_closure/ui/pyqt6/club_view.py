"""3D animated clubhead view and the rate-sweep plot.

The 3D view draws a simplified driver head (face plate, crown outline,
shaft stub) at impact orientation and animates its rotation under the
scenario's angular velocity across a few milliseconds either side of
impact, with the reference-point and impact-point velocity vectors drawn
to scale. Matplotlib 3D embedded in Qt, the house pattern for in-window
3D rendering.

Playback is user-controllable: play/pause, a 0.1x-3x speed multiplier,
and two display modes — "Head Fixed in Place" (rotation only, easiest
to read) and "Head Moving Through Space" (the head also translates
along the target line at the delivery speed, showing the true motion).
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from rate_of_closure.model import ImpactScenario, solve, sweep

logger = logging.getLogger(__name__)

__all__ = ["VIEW_MODES", "Club3DView", "SweepView"]

# Fallback palette (theme-neutral, matches shared CHART_COLORS hues).
_COL_FACE = "#0A84FF"
_COL_BODY = "#8b949e"
_COL_SHAFT = "#AC8E68"
_COL_V_REF = "#30D158"
_COL_V_POINT = "#FF375F"
_COL_IMPACT = "#FFD60A"
_COL_GROUND = "#8b949e"

# Simplified driver-head dimensions [m].
_FACE_HALF_WIDTH = 0.058
_FACE_HALF_HEIGHT = 0.028
_BODY_DEPTH = 0.11
_SHAFT_STUB = 0.35

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
        self._phase = 0.0
        self._speed = 1.0
        self._timer = QTimer(self)
        self._timer.setInterval(_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)

    # ── construction ────────────────────────────────────────────────
    def _build_playback_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 4, 4, 0)

        self._play_button = QPushButton("Pause")
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
        self._mode_combo.setToolTip(
            "Fixed: rotation only, easiest to read.\n"
            "Moving: the head also translates down the target line at the "
            "delivery speed."
        )
        self._mode_combo.currentTextChanged.connect(lambda _t: self._draw())
        bar.addWidget(self._mode_combo)
        return bar

    # ── public API ──────────────────────────────────────────────────
    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Adopt a new scenario and restart the rotation animation."""
        self._scenario = scenario
        self._phase = 0.0
        if not self._timer.isActive() and not self._play_button.isChecked():
            self._timer.start()
        self._draw()

    def set_playback_speed(self, multiplier: float) -> None:
        """Set the animation speed multiplier (0.1-3.0)."""
        clamped = max(0.1, min(3.0, multiplier))
        self._speed_slider.setValue(round(clamped * 100))

    def playback_speed(self) -> float:
        """Current animation speed multiplier."""
        return self._speed

    def set_view_mode(self, mode: str) -> None:
        """Select a display mode by name (see :data:`VIEW_MODES`)."""
        if mode not in VIEW_MODES:
            logger.warning("unknown view mode requested: %s", mode)
            return
        self._mode_combo.setCurrentText(mode)

    def view_mode(self) -> str:
        """The active display mode name."""
        return self._mode_combo.currentText()

    def stop(self) -> None:
        """Stop the animation timer (used on window close and in tests)."""
        self._timer.stop()

    # ── internals ──────────────────────────────────────────────────
    def _on_play_toggled(self, paused: bool) -> None:
        self._play_button.setText("Play" if paused else "Pause")
        if paused:
            self._timer.stop()
        else:
            self._timer.start()

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
        axes.clear()
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
                arrow_length_ratio=0.12,
                label=label,
            )

        limit = 0.24 if not moving else 0.42
        axes.set_xlim(-limit, limit)
        axes.set_ylim(-limit * 0.6, limit * 1.4)
        axes.set_zlim(-limit * 0.6, limit * 1.4)
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


class SweepView(QWidget):
    """Path deviation as a function of about-shaft rotation rate."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(5, 3.4), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._axes = self._figure.add_subplot(111)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Redraw the sweep for a new scenario."""
        rates = np.linspace(0.0, 4000.0, 81)
        deviations = sweep(scenario, "omega_shaft_dps", rates)
        current = solve(scenario)

        axes = self._axes
        axes.clear()
        axes.plot(rates, deviations, color=_COL_FACE, lw=2.0)
        axes.axvline(scenario.omega_shaft_dps, color=_COL_V_POINT, lw=1.0, ls="--")
        axes.scatter(
            [scenario.omega_shaft_dps],
            [current.path_deviation_deg],
            color=_COL_V_POINT,
            zorder=5,
        )
        axes.axhline(0.0, color=_COL_BODY, lw=0.8)
        axes.set_xlabel("About-Shaft Rotation Rate [deg/s]")
        axes.set_ylabel("Impact-Point Path Deviation [deg]")
        axes.grid(alpha=0.25)
        self._canvas.draw_idle()
