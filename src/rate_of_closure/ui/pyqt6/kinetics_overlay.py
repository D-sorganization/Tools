"""3D kinetics overlay geometry for the swing viewer (#4125 H2).

Pure-numpy helpers producing torque arcs and force arrows for one
playback frame of a :class:`~rate_of_closure.simulation.kinetics.
KineticsSeries`. The drawing pattern (arc sweep proportional to the
torque magnitude, direction by sign; force arrows auto-scaled with a
cap so the scene stays readable) is adapted from the movement
optimizer's canvas overlay (``src/movement_optimizer/gui/
vector_overlay.py`` — ``TorqueArc`` / ``ForceArrow`` /
``auto_scale_factor``), re-expressed for the app-frame 3D scene.

Sign convention: positive torque sweeps counter-clockwise in the swing
plane's (local x, local up) axes — the direction of increasing joint
angle (see ``rate_of_closure.simulation.kinetics``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rate_of_closure._contracts import require
from rate_of_closure.simulation.kinetics import KineticsSeries

__all__ = [
    "MAX_ARC_RADIUS_M",
    "MAX_ARC_SWEEP_DEG",
    "MAX_ARROW_LENGTH_M",
    "OverlayFrame",
    "overlay_frame",
]

#: Arc radius [m] at (or beyond) the reference torque magnitude.
MAX_ARC_RADIUS_M = 0.45
#: Minimum arc radius [m] so small torques stay visible.
MIN_ARC_RADIUS_M = 0.08
#: Arc sweep at full magnitude (movement-optimizer convention: 270°).
MAX_ARC_SWEEP_DEG = 270.0
#: Force arrows are scaled so the frame's largest arrow has this
#: length [m], capped (movement-optimizer ``auto_scale_factor``).
MAX_ARROW_LENGTH_M = 0.8
_ARC_POINTS = 24


@dataclass(frozen=True)
class OverlayFrame:
    """Drawable kinetics overlay for one playback frame.

    Attributes:
        arcs: ``(label, points)`` per joint — ``points`` is (K, 3)
            app-frame polyline tracing the torque arc; the label
            carries the joint name and signed torque [N·m].
        arrows: ``(label, start, vector)`` per force — app-frame arrow
            base and (already display-scaled) direction vector; the
            label carries the true magnitude [N].
    """

    arcs: tuple[tuple[str, np.ndarray], ...]
    arrows: tuple[tuple[str, np.ndarray, np.ndarray], ...] = field(repr=False)


def _arc_points(
    center: np.ndarray,
    torque_nm: float,
    reference_nm: float,
    x_axis: np.ndarray,
    up_axis: np.ndarray,
) -> np.ndarray:
    """(K, 3) polyline for one torque arc (radius ∝ |τ|, sign = sweep)."""
    fraction = max(-1.0, min(1.0, torque_nm / reference_nm))
    radius = MIN_ARC_RADIUS_M + (MAX_ARC_RADIUS_M - MIN_ARC_RADIUS_M) * abs(fraction)
    sweep = math.radians(MAX_ARC_SWEEP_DEG) * fraction
    angles = np.linspace(0.0, sweep, _ARC_POINTS)
    return np.asarray(
        center
        + radius * np.outer(np.cos(angles), x_axis)
        + radius * np.outer(np.sin(angles), up_axis)
    )


def overlay_frame(series: KineticsSeries, index: int) -> OverlayFrame:
    """The overlay geometry at one swing-sample index.

    Torque arcs use the NET joint torque (``torque_inertial_nm``, the
    plotted series); reference magnitudes are the series' own peaks so
    the largest arc always spans the full sweep. Force arrows show the
    shoulder and wrist reaction forces plus the clubhead force
    estimate, jointly scaled by the series' peak magnitude and capped
    at :data:`MAX_ARROW_LENGTH_M`.

    Args:
        series: The run's kinetics.
        index: Swing-sample index (clamped into range).

    Returns:
        The frame geometry with legend-ready labels.
    """
    require(isinstance(series, KineticsSeries), "series must be a KineticsSeries")
    n = series.t.shape[0]
    i = min(max(int(index), 0), n - 1)

    x_axis, up_axis = series.plane_x_app, series.plane_up_app
    centers = (series.pivot_position_m, series.wrist_positions_m[i])
    arcs = []
    for j, (name, center) in enumerate(zip(series.joint_names, centers, strict=True)):
        torque = float(series.torque_inertial_nm[i, j])
        reference = float(np.abs(series.torque_inertial_nm[:, j]).max())
        if reference < 1e-9:
            continue
        arcs.append(
            (
                f"{name} torque {torque:+.1f} N·m",
                _arc_points(np.asarray(center), torque, reference, x_axis, up_axis),
            )
        )

    forces = (
        ("shoulder", series.pivot_position_m, series.shoulder_force_n[i]),
        ("wrist", series.wrist_positions_m[i], series.wrist_force_n[i]),
        ("clubhead", series.clubhead_positions_m[i], series.clubhead_force_n[i]),
    )
    peak = max(
        float(series.force_magnitude_n(which).max())
        for which in ("shoulder", "wrist", "clubhead")
    )
    arrows = []
    if peak > 1e-9:
        scale = MAX_ARROW_LENGTH_M / peak
        for name, start, vector in forces:
            magnitude = float(np.linalg.norm(vector))
            if magnitude < 1e-9:
                continue
            arrows.append(
                (
                    f"{name} force {magnitude:.0f} N",
                    np.asarray(start, dtype=float),
                    np.asarray(vector, dtype=float) * scale,
                )
            )
    return OverlayFrame(arcs=tuple(arcs), arrows=tuple(arrows))
