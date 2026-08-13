"""Professional Matplotlib rendering for the exact impact inspection scene."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def _arrowhead(
    axes: Any,
    display: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: str,
) -> None:
    """Draw a camera-correct 3-D arrowhead without duplicating legend entries."""
    shown = display(np.vstack([start, end]))
    delta = shown[1] - shown[0]
    axes.quiver(
        shown[0, 0],
        shown[0, 1],
        shown[0, 2],
        delta[0],
        delta[1],
        delta[2],
        color=color,
        arrow_length_ratio=0.45,
        linewidth=1.8,
        normalize=False,
    )


def draw_impact_scene_3d(
    axes: Any,
    scene: ImpactScene,
    display: Callable[[np.ndarray], np.ndarray],
    chart_color: Callable[[int], str],
    visible_layers: frozenset[str] | None = None,
) -> None:
    """Draw the frame-explicit club geometry and velocity decomposition."""
    contact = np.asarray(scene.contact_point_m)
    shaft_point = np.asarray(scene.shaft_axis_point_m)
    shaft_axis = np.asarray(scene.shaft_axis_unit)
    face_normal = np.asarray(scene.face_normal_unit)
    leading_edge = np.asarray(scene.leading_edge_unit)
    face_up = np.cross(leading_edge, face_normal)
    face_center = np.asarray(scene.face_center_point_m)
    layers = (
        frozenset(
            {
                "face_normal",
                "face_center_travel",
                "dplane_normal",
                "spin_loft_sector",
            }
        )
        if visible_layers is None
        else visible_layers
    )

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
    if "face_normal" in layers:
        center_normal = np.asarray(scene.face_center_normal_unit)
        endpoint = face_center + 0.15 * center_normal
        _line(
            axes,
            display,
            np.vstack([face_center, endpoint]),
            color=chart_color(2),
            lw=2.6,
            label="Face-Center Normal",
        )
        _arrowhead(
            axes,
            display,
            face_center + 0.115 * center_normal,
            endpoint,
            color=chart_color(2),
        )
    if (
        "face_center_travel" in layers
        and scene.face_center_dplane.travel_direction_unit is not None
    ):
        travel = np.asarray(scene.face_center_dplane.travel_direction_unit)
        endpoint = face_center + 0.15 * travel
        _line(
            axes,
            display,
            np.vstack([face_center, endpoint]),
            color="#f59e0b",
            lw=2.6,
            label="Face-Center Travel",
        )
        _arrowhead(
            axes,
            display,
            face_center + 0.115 * travel,
            endpoint,
            color="#f59e0b",
        )
    if (
        "dplane_normal" in layers
        and scene.face_center_dplane.dplane_normal_unit is not None
    ):
        normal = np.asarray(scene.face_center_dplane.dplane_normal_unit)
        endpoint = face_center + 0.13 * normal
        _line(
            axes,
            display,
            np.vstack([face_center, endpoint]),
            color="#14b8a6",
            lw=2.0,
            label="D-Plane Normal",
        )
        _arrowhead(
            axes,
            display,
            face_center + 0.098 * normal,
            endpoint,
            color="#14b8a6",
        )
    if "spin_loft_sector" in layers and scene.spin_loft_sector_unit:
        sector = np.vstack(
            [
                face_center,
                *(
                    face_center + 0.12 * np.asarray(direction)
                    for direction in scene.spin_loft_sector_unit
                ),
            ]
        )
        shown_sector = display(sector)
        collection = Poly3DCollection(
            [shown_sector],
            facecolors="#22d3ee",
            edgecolors="#67e8f9",
            alpha=0.22,
            linewidths=0.8,
            label="3D Spin-Loft Sector",
        )
        axes.add_collection3d(collection)

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
