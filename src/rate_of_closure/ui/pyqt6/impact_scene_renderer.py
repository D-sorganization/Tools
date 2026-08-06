"""Professional Matplotlib rendering for the exact impact inspection scene."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from rate_of_closure.simulation import ImpactScene

__all__ = ["draw_impact_scene_3d"]

_VECTOR_COLORS = {
    "total": 1,
    "axis_translation": 0,
    "shaft_rotation": 3,
    "other_rotation": 6,
    "without_shaft": 8,
}


def _line(
    axes: Any,
    display: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    **style: object,
) -> None:
    shown = display(points)
    axes.plot(shown[:, 0], shown[:, 1], shown[:, 2], **style)


def draw_impact_scene_3d(
    axes: Any,
    scene: ImpactScene,
    display: Callable[[np.ndarray], np.ndarray],
    chart_color: Callable[[int], str],
) -> None:
    """Draw the frame-explicit club geometry and velocity decomposition."""
    contact = np.asarray(scene.contact_point_m)
    shaft_point = np.asarray(scene.shaft_axis_point_m)
    shaft_axis = np.asarray(scene.shaft_axis_unit)
    face_normal = np.asarray(scene.face_normal_unit)
    leading_edge = np.asarray(scene.leading_edge_unit)
    face_up = np.cross(leading_edge, face_normal)

    _line(
        axes,
        display,
        np.vstack([shaft_point - 0.08 * shaft_axis, shaft_point + 0.45 * shaft_axis]),
        color=chart_color(7),
        lw=3.0,
        label="Physical Shaft Axis",
    )
    face_corners = np.asarray(
        [
            contact + edge * 0.06 + up * 0.035
            for edge, up in (
                (-leading_edge, -face_up),
                (leading_edge, -face_up),
                (leading_edge, face_up),
                (-leading_edge, face_up),
                (-leading_edge, -face_up),
            )
        ]
    )
    _line(
        axes,
        display,
        face_corners,
        color=chart_color(0),
        lw=2.0,
        label="Wedge Face",
    )
    _line(
        axes,
        display,
        np.vstack([contact - 0.06 * leading_edge, contact + 0.06 * leading_edge]),
        color=chart_color(4),
        lw=3.0,
        label="Leading Edge",
    )
    for label, direction, color in (
        ("Face Normal", face_normal, chart_color(2)),
        ("Arc Tangent", np.asarray(scene.arc_tangent_unit), chart_color(5)),
    ):
        _line(
            axes,
            display,
            np.vstack([contact, contact + 0.15 * direction]),
            color=color,
            lw=2.0,
            label=label,
        )

    max_speed = max(float(np.linalg.norm(vector.vector)) for vector in scene.vectors)
    arrow_scale = 0.18 / max(max_speed, 1e-12)
    for vector in scene.vectors:
        origin = np.asarray(vector.origin_m)
        tip = origin + arrow_scale * np.asarray(vector.vector)
        _line(
            axes,
            display,
            np.vstack([origin, tip]),
            color=chart_color(_VECTOR_COLORS[vector.key]),
            lw=2.2 if vector.key == "total" else 1.4,
            ls="--" if vector.key == "without_shaft" else "-",
            label=vector.label,
        )

    if scene.screw_axis is not None:
        screw_point = np.asarray(scene.screw_axis.point_m)
        screw_axis = np.asarray(scene.screw_axis.direction_unit)
        _line(
            axes,
            display,
            np.vstack([screw_point - 0.3 * screw_axis, screw_point + 0.3 * screw_axis]),
            color=chart_color(6),
            lw=1.8,
            ls=":",
            label="Instantaneous Screw Axis",
        )
    shown_contact = display(contact)
    axes.scatter(
        *shown_contact,
        marker="o",
        facecolors="none",
        edgecolors=chart_color(9),
        linewidths=2.0,
        s=95,
        label="Declared Contact Point",
        zorder=10,
    )
