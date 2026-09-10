"""Lens-corrected ruler measurements through the canonical triangulation API."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from .calibration import DistortionCoefficients, DistortionModel, PinholeIntrinsics
from .enums import Availability
from .extrinsics import CameraPose
from .observations import PixelObservation
from .reconstruction import (
    KeypointReconstruction,
    ReconstructionConfig,
    triangulate_n_views,
)
from .reference_placements import PlacementObservation, ReferenceCamera, ReferenceTarget
from .reference_scale import ReferenceScaleOptions, ReferenceScaleProblem, ScaleResidual

_MAX_CAMERAS = 12
_MAX_PLACEMENTS = 32


def _check_problem(problem: ReferenceScaleProblem) -> None:
    layout = problem.layout
    if layout.world_frame.length_unit != "m":
        raise ValueError("camera layout must use metre units")
    if not 2 <= len(problem.cameras) <= _MAX_CAMERAS:
        raise ValueError("scale measurements need 2–12 established cameras")
    if set(layout.camera_poses) != set(problem.cameras):
        raise ValueError("camera profiles must match every layout camera")
    for key, camera in problem.cameras.items():
        if not isinstance(camera, ReferenceCamera) or camera.camera_key != key:
            raise ValueError("camera profile identity does not match its key")
        if not isinstance(camera.intrinsics, PinholeIntrinsics):
            raise ValueError("linear scale currently requires pinhole lens profiles")
        pose = layout.get_pose(key)
        if pose.t_camera_from_world.source_frame_id != layout.world_frame.frame_id:
            raise ValueError("camera transform must map from the layout world frame")
    for key, target in problem.targets.items():
        if not isinstance(target, ReferenceTarget) or target.reference_id != key:
            raise ValueError("reference identity does not match its key")
        if len(target.point_ids) != 2:
            raise ValueError("linear scale needs two identified ruler endpoints")
        length = np.linalg.norm(np.diff(np.asarray(target.object_points_m), axis=0))
        if not np.isfinite(length) or length <= 0:
            raise ValueError("reference length must be finite and positive")


def _check_row(problem: ReferenceScaleProblem, row: PlacementObservation) -> None:
    if not isinstance(row, PlacementObservation):
        raise TypeError("observations must be PlacementObservation records")
    if row.camera_key not in problem.cameras or row.reference_id not in problem.targets:
        raise ValueError("observation references an unknown camera or ruler")
    camera = problem.cameras[row.camera_key]
    target = problem.targets[row.reference_id]
    if row.profile_id != camera.profile_id:
        raise ValueError("observation uses a different lens/zoom profile")
    if set(row.point_ids) != set(target.point_ids):
        raise ValueError("both identified ruler endpoints are required")
    pixels = np.asarray(row.pixels_px)
    if np.any(pixels < 0) or np.any(pixels >= camera.intrinsics.resolution_px):
        raise ValueError("reference pixels must lie within the recorded image")


def _groups(problem: ReferenceScaleProblem) -> dict[str, list[PlacementObservation]]:
    _check_problem(problem)
    if len(problem.observations) > _MAX_CAMERAS * _MAX_PLACEMENTS:
        raise ValueError("too many ruler observations")
    groups: dict[str, list[PlacementObservation]] = {}
    seen: set[tuple[str, str]] = set()
    for row in problem.observations:
        _check_row(problem, row)
        key = (row.placement_id, row.camera_key)
        if key in seen:
            raise ValueError("duplicate camera observation for a ruler placement")
        seen.add(key)
        group = groups.setdefault(row.placement_id, [])
        if group and (row.reference_id, row.held_out) != (
            group[0].reference_id,
            group[0].held_out,
        ):
            raise ValueError("a placement must share its reference and holdout status")
        group.append(row)
    if not 1 <= len(groups) <= _MAX_PLACEMENTS:
        raise ValueError("use 1–32 distinct ruler placements")
    if not any(not rows[0].held_out for rows in groups.values()):
        raise ValueError("at least one fitting placement must not be held out")
    signatures = set()
    for rows in groups.values():
        if len(rows) < 2:
            raise ValueError("each ruler placement needs at least two camera views")
        signature = tuple(
            sorted((row.camera_key, tuple(sorted(row.pixels_px))) for row in rows)
        )
        if signature in signatures:
            raise ValueError("identical ruler images are not independent placements")
        signatures.add(signature)
    return groups


def _pixel(
    row: PlacementObservation,
    point_id: str,
    camera: ReferenceCamera,
    ideal: PinholeIntrinsics,
) -> PixelObservation:
    raw = row.pixels_px[row.point_ids.index(point_id)]
    ray = camera.intrinsics.unproject_point(raw)
    return PixelObservation(
        observation_id=f"{row.placement_id}:{row.camera_key}:{point_id}",
        camera_id=row.camera_key,
        frame_sequence=row.frame_sequence,
        timestamp_ns=row.timestamp_ns,
        skeleton_id="linear-reference",
        keypoint_id=point_id,
        uv_px=ideal.project_point(ray),
        confidence=1.0,
        covariance_px2=(1.0, 0.0, 0.0, 1.0),
        availability=Availability.OBSERVED,
    )


def _rotation(pose: CameraPose) -> Rotation:
    scalar, x, y, z = pose.t_camera_from_world.rotation_wxyz
    return Rotation.from_quat((x, y, z, scalar))


def _quality(
    problem: ReferenceScaleProblem,
    rows: list[PlacementObservation],
    point_id: str,
    result: KeypointReconstruction,
) -> tuple[float, float]:
    point = np.asarray(result.landmark.xyz_m)
    centers, errors = [], []
    for row in rows:
        if row.camera_key not in result.inlier_camera_ids:
            continue
        pose = problem.layout.get_pose(row.camera_key)
        centers.append(pose.camera_center_world)
        camera_point = (
            _rotation(pose).apply(point) + pose.t_camera_from_world.translation_m
        )
        lens = problem.cameras[row.camera_key].intrinsics
        pixel = lens.project_point(tuple(camera_point))
        raw = row.pixels_px[row.point_ids.index(point_id)]
        errors.append(float(np.linalg.norm(np.asarray(pixel) - raw)))
    rays = point - np.asarray(centers)
    lengths = np.linalg.norm(rays, axis=1)
    if np.any(lengths <= np.finfo(float).eps):
        raise ValueError("reference point coincides with a camera center")
    rays /= lengths[:, None]
    # Antiparallel rays are just as degenerate as parallel rays.
    dot = np.abs(rays @ rays.T)
    pairs = dot[np.triu_indices(len(rays), 1)]
    angle = float(np.degrees(np.arccos(np.clip(pairs, 0, 1))).max())
    return angle, max(errors)


def _ideal_lens(camera: ReferenceCamera) -> PinholeIntrinsics:
    lens = camera.intrinsics
    if not isinstance(lens, PinholeIntrinsics):
        raise ValueError("linear scale requires a pinhole lens profile")
    return replace(lens, distortion=DistortionCoefficients(DistortionModel.NONE, ()))


def _endpoint(
    problem: ReferenceScaleProblem,
    rows: list[PlacementObservation],
    point_id: str,
    options: ReferenceScaleOptions,
) -> tuple[NDArray[np.float64], float, float]:
    ideal = {key: _ideal_lens(camera) for key, camera in problem.cameras.items()}
    pixels = [
        _pixel(row, point_id, problem.cameras[row.camera_key], ideal[row.camera_key])
        for row in rows
    ]
    config = ReconstructionConfig(
        max_reprojection_error_px=options.max_reprojection_error_px
    )
    result = triangulate_n_views(pixels, problem.layout, ideal, config)
    if (
        result.landmark.availability is not Availability.DERIVED
        or len(result.inlier_camera_ids) < 2
    ):
        raise ValueError("ruler endpoint has insufficient triangulation evidence")
    angle, error = _quality(problem, rows, point_id, result)
    if angle < options.min_parallax_degrees:
        raise ValueError("ruler triangulation has insufficient baseline/parallax")
    if error > options.max_reprojection_error_px:
        raise ValueError("ruler endpoint exceeds raw-image reprojection tolerance")
    return np.asarray(result.landmark.xyz_m), angle, error


def measure_placements(
    problem: ReferenceScaleProblem, options: ReferenceScaleOptions
) -> tuple[ScaleResidual, ...]:
    """Measure endpoints without fitting holdout lengths or changing lenses."""
    groups = _groups(problem)
    measured = []
    for placement, rows in sorted(groups.items()):
        target = problem.targets[rows[0].reference_id]
        endpoints = [
            _endpoint(problem, rows, point, options) for point in target.point_ids
        ]
        length = float(np.linalg.norm(endpoints[0][0] - endpoints[1][0]))
        known = float(
            np.linalg.norm(np.diff(np.asarray(target.object_points_m), axis=0))
        )
        if not np.isfinite(length) or length <= np.finfo(float).eps:
            raise ValueError("reconstructed ruler length must be finite and positive")
        measured.append(
            ScaleResidual(
                placement,
                known,
                length,
                length,
                (length - known) / known,
                rows[0].held_out,
                min(point[1] for point in endpoints),
                max(point[2] for point in endpoints),
            )
        )
    return tuple(measured)
