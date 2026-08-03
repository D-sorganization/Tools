"""3D animated clubhead view and the rate-sweep plot.

The 3D view draws a simplified driver head (face plate, crown outline,
shaft stub) at impact orientation and animates its rotation under the
scenario's angular velocity across a few milliseconds either side of
impact, with the reference-point and impact-point velocity vectors drawn
to scale. Matplotlib 3D embedded in Qt, the house pattern for in-window
3D rendering.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from rate_of_closure.model import ImpactScenario, solve, sweep

logger = logging.getLogger(__name__)

__all__ = ["Club3DView", "SweepView"]

# Fallback palette (theme-neutral, matches shared CHART_COLORS hues).
_COL_FACE = "#0A84FF"
_COL_BODY = "#8b949e"
_COL_SHAFT = "#AC8E68"
_COL_V_REF = "#30D158"
_COL_V_POINT = "#FF375F"
_COL_IMPACT = "#FFD60A"

# Simplified driver-head dimensions [m].
_FACE_HALF_WIDTH = 0.058
_FACE_HALF_HEIGHT = 0.028
_BODY_DEPTH = 0.11
_SHAFT_STUB = 0.35

_ANIMATION_SPAN_MS = 8.0
_ANIMATION_STEPS = 48


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
        layout.addWidget(self._canvas)

        self._scenario: ImpactScenario | None = None
        self._step = 0
        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._advance)

    def set_scenario(self, scenario: ImpactScenario) -> None:
        """Adopt a new scenario and restart the rotation animation."""
        self._scenario = scenario
        self._step = 0
        if not self._timer.isActive():
            self._timer.start()
        self._draw()

    def stop(self) -> None:
        """Stop the animation timer (used on window close and in tests)."""
        self._timer.stop()

    # ── internals ──────────────────────────────────────────────────
    def _advance(self) -> None:
        self._step = (self._step + 1) % _ANIMATION_STEPS
        self._draw()

    def _draw(self) -> None:
        if self._scenario is None:
            return
        scenario = self._scenario
        result = solve(scenario)
        omega = np.radians(np.array(result.omega_dps))
        phase = self._step / (_ANIMATION_STEPS - 1) - 0.5
        rotation = _rodrigues(omega, phase * _ANIMATION_SPAN_MS / 1000.0)

        parts = _head_wireframe(scenario)
        axes = self._axes
        axes.clear()
        for key, color, width in (
            ("face", _COL_FACE, 2.2),
            ("back", _COL_BODY, 1.2),
            ("shaft", _COL_SHAFT, 2.0),
        ):
            pts = _display(parts[key] @ rotation.T)
            axes.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=width)
        for a, b in zip(parts["face"], parts["back"], strict=True):
            seg = _display(np.vstack([a, b]) @ rotation.T)
            axes.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=_COL_BODY, lw=0.8)

        impact = parts["impact"] @ rotation.T
        axes.scatter(*_display(impact), color=_COL_IMPACT, s=45, zorder=5)
        axes.scatter(0.0, 0.0, 0.0, color=_COL_BODY, s=30)

        scale = 0.0035  # m per (m/s): keeps arrows inside the box
        v_ref = np.array([result.reference_speed_mph, 0.0, 0.0]) * 0.44704
        v_point = np.array(result.point_velocity_mps)
        for origin, vec, color, label in (
            (np.zeros(3), v_ref, _COL_V_REF, "reference (GC) path"),
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

        limit = 0.24
        axes.set_xlim(-limit, limit)
        axes.set_ylim(-limit * 0.6, limit * 1.4)
        axes.set_zlim(-limit * 0.6, limit * 1.4)
        axes.set_xlabel("z — right of target [m]")
        axes.set_ylabel("x — target line [m]")
        axes.set_zlabel("y — up [m]")
        axes.set_title(
            f"Path Δ {result.path_deviation_deg:+.2f}°   "
            f"AoA Δ {result.aoa_deviation_deg:+.2f}°"
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
        axes.set_xlabel("About-shaft rotation rate [deg/s]")
        axes.set_ylabel("Impact-point path deviation [deg]")
        axes.grid(alpha=0.25)
        self._canvas.draw_idle()
