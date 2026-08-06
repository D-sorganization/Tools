"""Engineering presentation helpers for instantaneous screw motion."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rate_of_closure.simulation.screw_analysis import (
    MotionKind,
    ScrewMotion,
    build_screw_glyph,
    project_motion,
)


def _velocity_angles(velocity: np.ndarray) -> tuple[float, float] | None:
    """Return exact (angle of attack, path) angles, or ``None`` at rest."""
    if float(np.linalg.norm(velocity)) < 1e-10:
        return None
    attack = np.degrees(np.arctan2(velocity[1], np.hypot(velocity[0], velocity[2])))
    path = np.degrees(np.arctan2(velocity[2], velocity[0]))
    return float(attack), float(path)


def format_screw_readout(
    label: str,
    motion: ScrewMotion,
    loft_deg: float,
    contribution_residual_m_s: float | None = None,
) -> str:
    """Describe one screw-motion glyph, decomposition, and output projections."""
    if motion.kind is not MotionKind.FINITE:
        description = (
            "Pure translation: the screw axis is at infinity; the arrow shows "
            "the translation direction."
            if motion.kind is MotionKind.TRANSLATION
            else "Stationary: angular and reference-point speeds are zero, so no "
            "axis exists."
        )
        return f"{label} — {description}"

    loft = np.radians(loft_deg)
    directions = {
        "target": np.array([1.0, 0.0, 0.0]),
        "vertical": np.array([0.0, 1.0, 0.0]),
        "lateral": np.array([0.0, 0.0, 1.0]),
        "face_normal": np.array([np.cos(loft), np.sin(loft), 0.0]),
    }
    projections = project_motion(motion, directions)
    angles = _velocity_angles(motion.reference_velocity_m_s)
    angle_text = (
        f" AoA {angles[0]:+.2f}°, Path {angles[1]:+.2f}°; "
        if angles is not None
        else " AoA/Path undefined at zero speed; "
    )
    residual_text = (
        f" · Joint Contribution Residual {contribution_residual_m_s:.3e} m/s"
        if contribution_residual_m_s is not None
        else ""
    )
    projection_text = "; ".join(
        f"{display_name} {projection.total_m_s:+.3f} = "
        f"{projection.orbital_m_s:+.3f} orbital + "
        f"{projection.axial_m_s:+.3f} axial"
        for display_name, projection in (
            ("Target", projections["target"]),
            ("Vertical/AoA", projections["vertical"]),
            ("Lateral/Path", projections["lateral"]),
            ("Face Normal", projections["face_normal"]),
        )
    )
    return (
        f"{label} — Finite screw in app/world frame (x target, y up, z right). "
        "Axis arrow follows ω; wrapped arc shows rotation handedness. "
        f"ω {np.degrees(motion.angular_rate_rad_s):.1f} °/s · "
        f"Pitch {motion.pitch_m_rad:+.4f} m/rad · "
        f"Axial {motion.axial_speed_m_s:+.3f} m/s · R_ISA {motion.radius_m:.3f} m. "
        "Orbital + axial velocity reconstructs the selected-point velocity. "
        "Signed projection breakdown [total = orbital + axial] (m/s): "
        f"{projection_text}."
        f"{angle_text}orbital/axial direction angles are diagnostics, not additive"
        f"{residual_text}."
    )


@dataclass(frozen=True)
class ScrewOverlayRenderer:
    """Render a screw glyph through a scene's coordinate presentation adapter."""

    axes: Any
    display: Callable[[np.ndarray], np.ndarray]
    chart_color: Callable[[int], str]

    def draw(self, motion: ScrewMotion, extent: float, label: str) -> None:
        """Draw a directed axis, wrapped helix, reference radius, or translation."""
        glyph = build_screw_glyph(motion, extent)
        if glyph is None:
            self._draw_degenerate(motion, extent)
            return

        axis = self.display(glyph.axis_line_m)
        helix = self.display(glyph.helix_m)
        radius = self.display(glyph.radius_line_m)
        color = self.chart_color(5)
        self.axes.plot(*axis.T, color=color, lw=2.4, label=f"Screw Axis — {label}")
        self.axes.plot(
            *helix.T,
            color=self.chart_color(3),
            lw=2.0,
            label="Helical Motion — rotation handedness",
        )
        self.axes.plot(
            *radius.T,
            color=self.chart_color(6),
            lw=1.4,
            ls=":",
            label="R_ISA — reference-point radius",
        )
        direction = self.display(motion.axis_direction) * extent * 0.22
        start = self.display(
            glyph.axis_line_m[1] - motion.axis_direction * extent * 0.22
        )
        self.axes.quiver(*start, *direction, color=color, arrow_length_ratio=0.32)

    def _draw_degenerate(self, motion: ScrewMotion, extent: float) -> None:
        """Truthfully render translation direction or a stationary point."""
        if motion.kind is MotionKind.STATIONARY:
            return
        start = self.display(motion.reference_point_m)
        direction = self.display(motion.axis_direction) * extent * 0.7
        self.axes.quiver(
            *start,
            *direction,
            color=self.chart_color(5),
            arrow_length_ratio=0.18,
            label="Pure Translation — Screw Axis at Infinity",
        )
