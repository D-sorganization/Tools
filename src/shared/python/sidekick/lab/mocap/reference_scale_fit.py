"""One-parameter length fit and immutable anchored camera-layout rescaling."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from uuid import uuid4

import numpy as np

from .extrinsics import CameraLayout, CameraPose
from .reference_scale import (
    ReferenceScaleOptions,
    ReferenceScaleProblem,
    ReferenceScaleResult,
    ScaleResidual,
)
from .reference_scale_triangulation import _rotation


def _scaled_pose(pose: CameraPose, scale: float, anchor: np.ndarray) -> CameraPose:
    center = anchor + scale * (np.asarray(pose.camera_center_world) - anchor)
    translation = -_rotation(pose).apply(center)
    transform = replace(
        pose.t_camera_from_world,
        translation_m=(
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ),
    )
    return replace(pose, t_camera_from_world=transform)


def _layout(
    problem: ReferenceScaleProblem, scale: float, options: ReferenceScaleOptions
) -> CameraLayout:
    anchor = np.asarray(options.anchor_m)
    poses = {
        key: _scaled_pose(pose, scale, anchor)
        for key, pose in problem.layout.camera_poses.items()
    }
    layout = CameraLayout(str(uuid4()), problem.layout.world_frame, poses)
    object.__setattr__(
        layout, "camera_poses", MappingProxyType(dict(layout.camera_poses))
    )
    return layout


def _factor(fitting: tuple[ScaleResidual, ...]) -> tuple[float, float | None]:
    """Minimize sum((scale * reconstructed_length - measured_length)**2).

    Equal weights avoid interpreting manual point confidence as calibrated noise.
    Held-out placements are evaluated after fitting and never enter the objective.
    """
    reconstructed = np.asarray([row.reconstructed_length_m for row in fitting])
    known = np.asarray([row.known_length_m for row in fitting])
    scale = float(np.dot(reconstructed, known) / np.dot(reconstructed, reconstructed))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("reference evidence does not give a finite positive scale")
    repeatability = (
        float(np.std(known / reconstructed, ddof=1) / np.sqrt(len(fitting)))
        if len(fitting) > 1
        else None
    )
    return scale, repeatability


def _residuals(
    measurements: tuple[ScaleResidual, ...], scale: float, limit: float
) -> tuple[ScaleResidual, ...]:
    residuals = tuple(
        replace(
            row,
            scaled_length_m=scale * row.reconstructed_length_m,
            relative_error=scale * row.reconstructed_length_m / row.known_length_m - 1,
        )
        for row in measurements
    )
    bad = [row.placement_id for row in residuals if abs(row.relative_error) > limit]
    if bad:
        raise ValueError(
            f"inconsistent ruler length evidence exceeds tolerance: {', '.join(bad)}"
        )
    return residuals


def _limitations(
    repeatability: float | None, holdouts: tuple[str, ...]
) -> tuple[str, ...]:
    limitations = [
        "Relative camera geometry and lens profiles are assumed correct; "
        "only global scale is estimated.",
        "Repeatability excludes lens, reference-length and pixel-picking uncertainty; "
        "it is not physical accuracy approval.",
    ]
    if repeatability is None:
        limitations.append("One fitting placement cannot establish repeatability.")
    if not holdouts:
        limitations.append("No independent held-out placement validates this scale.")
    return tuple(limitations)


def fit_scale(
    problem: ReferenceScaleProblem,
    measurements: tuple[ScaleResidual, ...],
    options: ReferenceScaleOptions,
) -> ReferenceScaleResult:
    """Return a reviewable candidate after fitting and independent length checks."""
    fitting = tuple(row for row in measurements if not row.held_out)
    scale, repeatability = _factor(fitting)
    residuals = _residuals(measurements, scale, options.max_relative_error)
    holdouts = tuple(row.placement_id for row in residuals if row.held_out)
    return ReferenceScaleResult(
        source_layout_id=problem.layout.layout_id,
        scaled_layout=_layout(problem, scale, options),
        scale_factor=scale,
        fit_placement_ids=tuple(row.placement_id for row in fitting),
        held_out_placement_ids=holdouts,
        residuals=residuals,
        scale_repeatability=repeatability,
        camera_profiles={
            key: camera.profile_id for key, camera in problem.cameras.items()
        },
        limitations=_limitations(repeatability, holdouts),
    )
