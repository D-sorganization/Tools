"""Dependency-light geometry adapters for the PyQt6 clubhead viewport."""

from __future__ import annotations

from typing import cast

import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from rate_of_closure.application.camera_presets import CameraViewId
from rate_of_closure.model import ImpactScenario

SHAFT_STUB_M = 0.35
_FACE_HALF_WIDTH_M = 0.058
_FACE_HALF_HEIGHT_M = 0.028
_BODY_DEPTH_M = 0.11


def rodrigues(axis_omega: np.ndarray, dt: float) -> np.ndarray:
    """Return the rotation matrix for ``axis_omega`` [rad/s] over ``dt`` s."""
    theta = float(np.linalg.norm(axis_omega)) * dt
    if abs(theta) < 1e-12:
        return cast(np.ndarray, np.eye(3))
    axis = axis_omega / np.linalg.norm(axis_omega)
    cross_matrix = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    rotation = (
        np.eye(3)
        + np.sin(theta) * cross_matrix
        + (1.0 - np.cos(theta)) * (cross_matrix @ cross_matrix)
    )
    return cast(np.ndarray, rotation)


def head_wireframe(scenario: ImpactScenario) -> dict[str, np.ndarray]:
    """Return head line strips in app frame x=downrange, y=up, z=right."""
    face_depth = scenario.com_to_face_mm / 1000.0
    width, height = _FACE_HALF_WIDTH_M, _FACE_HALF_HEIGHT_M
    face = np.array(
        [
            [face_depth, -height, -width],
            [face_depth, -height, width],
            [face_depth, height, width],
            [face_depth, height, -width],
            [face_depth, -height, -width],
        ]
    )
    back = face - np.array([_BODY_DEPTH_M, 0.0, 0.0])
    shaft_direction = np.array(
        [
            0.0,
            np.sin(np.radians(scenario.lie_angle_deg)),
            -np.cos(np.radians(scenario.lie_angle_deg)),
        ]
    )
    hosel = np.array([face_depth - 0.02, height, -width])
    shaft = np.vstack([hosel, hosel + shaft_direction * SHAFT_STUB_M])
    impact = np.array(
        [
            face_depth,
            scenario.impact_offset_high_mm / 1000.0,
            scenario.impact_offset_toe_mm / 1000.0,
        ]
    )
    return {"face": face, "back": back, "shaft": shaft, "impact": impact}


def display_points(points: np.ndarray) -> np.ndarray:
    """Map app-frame points to Matplotlib display axes ``(z, x, y)``."""
    return np.asarray(points)[..., [2, 0, 1]]


def canonical_axis_visibility(
    view_id: CameraViewId | None,
) -> tuple[bool, bool, bool]:
    """Return display-axis visibility, hiding only a canonical depth axis."""
    if view_id is CameraViewId.FACE_ON:
        return False, True, True
    if view_id is CameraViewId.DOWN_THE_LINE:
        return True, False, True
    if view_id is CameraViewId.OVERHEAD:
        return True, True, False
    return True, True, True


def set_axis_visibility(axes: Axes3D, view_id: CameraViewId | None) -> None:
    """Apply canonical depth-axis visibility to a Matplotlib 3D axes."""
    for axis, visible in zip(
        (axes.xaxis, axes.yaxis, axes.zaxis),
        canonical_axis_visibility(view_id),
        strict=True,
    ):
        axis.set_visible(visible)
        axis.pane.set_visible(visible)
        axis.gridlines.set_visible(visible)
        axis.label.set_visible(visible)
        if not visible:
            axis.set_ticks([])
            axis.set_label_text("")


__all__ = [
    "SHAFT_STUB_M",
    "canonical_axis_visibility",
    "display_points",
    "head_wireframe",
    "rodrigues",
    "set_axis_visibility",
]
