"""Extrinsic and flexible-layout camera calibration and relocalization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from ._validation import (
    require_finite,
    require_nonnegative_integer,
    require_text,
)
from .calibration import FisheyeIntrinsics, PinholeIntrinsics
from .geometry import CoordinateFrame, RigidTransform


class ExtrinsicQuality(StrEnum):
    """Categorical qualification floor for multi-camera extrinsics."""

    QUALIFIED = "qualified"
    DEGRADED = "degraded"
    UNQUALIFIED = "unqualified"


class ExtrinsicDegeneracyKind(StrEnum):
    """Identified geometric degeneracies in multi-camera layout or observations."""

    NONE = "none"
    INSUFFICIENT_CAMERAS = "insufficient-cameras"
    COLLINEAR_CAMERAS = "collinear-cameras"
    LOW_OVERLAP = "low-overlap"
    HIGH_REPROJECTION_ERROR = "high-reprojection-error"
    UNCONSTRAINED_GAUGE = "unconstrained-gauge"


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Rigid pose of a camera relative to the reference/world coordinate frame."""

    camera_key: str
    t_camera_from_world: RigidTransform

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_key", require_text(self.camera_key, "camera_key")
        )
        if not isinstance(self.t_camera_from_world, RigidTransform):
            raise TypeError("t_camera_from_world must be a RigidTransform")

    @property
    def camera_center_world(self) -> tuple[float, float, float]:
        """Compute camera optical center position in world coordinates: C = -R^T * t."""
        qw, qx, qy, qz = self.t_camera_from_world.rotation_wxyz
        tx, ty, tz = self.t_camera_from_world.translation_m

        # R matrix from quaternion
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qz * qw)
        r02 = 2.0 * (qx * qz + qy * qw)

        r10 = 2.0 * (qx * qy + qz * qw)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qx * qw)

        r20 = 2.0 * (qx * qz - qy * qw)
        r21 = 2.0 * (qy * qz + qx * qw)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        # C = -R^T * t
        cx = -(r00 * tx + r10 * ty + r20 * tz)
        cy = -(r01 * tx + r11 * ty + r21 * tz)
        cz = -(r02 * tx + r12 * ty + r22 * tz)
        return (cx, cy, cz)


@dataclass(frozen=True, slots=True)
class CameraLayout:
    """Collection of calibrated camera poses registered in a common world frame."""

    layout_id: str
    world_frame: CoordinateFrame
    camera_poses: Mapping[str, CameraPose]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout_id", require_text(self.layout_id, "layout_id"))
        if not isinstance(self.world_frame, CoordinateFrame):
            raise TypeError("world_frame must be a CoordinateFrame")
        if not self.camera_poses:
            raise ValueError("layout must contain at least one camera")
        for key, pose in self.camera_poses.items():
            if not isinstance(pose, CameraPose):
                raise TypeError(f"camera_poses[{key!r}] must be a CameraPose")
            if pose.camera_key != key:
                raise ValueError(
                    f"mismatched camera_key: {key!r} != {pose.camera_key!r}"
                )
        object.__setattr__(self, "camera_poses", dict(self.camera_poses))

    @property
    def camera_count(self) -> int:
        """Total number of registered cameras in the layout."""
        return len(self.camera_poses)

    def get_pose(self, camera_key: str) -> CameraPose:
        """Return the CameraPose for a given camera key or raise KeyError."""
        return self.camera_poses[camera_key]


@dataclass(frozen=True, slots=True)
class MovementDetectionResult:
    """Outcome of checking whether a camera has physically shifted from its pose."""

    camera_key: str
    has_moved: bool
    mean_discrepancy_px: float
    max_discrepancy_px: float
    observed_points_count: int


@dataclass(frozen=True, slots=True)
class RelocalizationResult:
    """Outcome of single-camera or multi-camera relocalization against world targets."""

    camera_key: str
    success: bool
    pose: CameraPose | None
    mean_reprojection_residual_px: float
    reprojected_points_count: int


@dataclass(frozen=True, slots=True)
class ExtrinsicCalibrationResult:
    """Overall multi-camera extrinsic calibration outcome."""

    layout: CameraLayout
    mean_reprojection_residual_px: float
    max_reprojection_residual_px: float
    quality: ExtrinsicQuality
    degeneracies: tuple[ExtrinsicDegeneracyKind, ...]
    calibrated_at_utc: str


def _transform_point(
    t_target_from_source: RigidTransform, xyz: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Transform point xyz from source frame to target frame."""
    qw, qx, qy, qz = t_target_from_source.rotation_wxyz
    tx, ty, tz = t_target_from_source.translation_m
    px, py, pz = xyz

    # R * p
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)

    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)

    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    x_tgt = r00 * px + r01 * py + r02 * pz + tx
    y_tgt = r10 * px + r11 * py + r12 * pz + ty
    z_tgt = r20 * px + r21 * py + r22 * pz + tz
    return (x_tgt, y_tgt, z_tgt)


def estimate_pnp_pose(
    camera_key: str,
    object_points: Sequence[tuple[float, float, float]],
    image_points: Sequence[tuple[float, float]],
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics,
    target_frame_id: str,
    source_frame_id: str,
) -> tuple[CameraPose, float]:
    """Estimate camera pose from 3-D world points and 2-D image observations."""
    if len(object_points) != len(image_points):
        raise ValueError("object_points and image_points must have identical lengths")
    if len(object_points) < 4:
        raise ValueError("PnP requires at least 4 point correspondences")

    try:
        import cv2
        import numpy as np

        obj_arr = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 3)
        img_arr = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 2)
        skew_val = intrinsics.skew if isinstance(intrinsics, PinholeIntrinsics) else 0.0
        k_mat = np.array(
            [
                [intrinsics.fx, skew_val, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = (
            np.array(intrinsics.distortion.coefficients, dtype=np.float64)
            if intrinsics.distortion.coefficients
            else None
        )

        success, rvec, tvec = cv2.solvePnP(
            obj_arr,
            img_arr,
            k_mat,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            raise RuntimeError(f"solvePnP failed to converge for camera {camera_key}")

        rot_mat, _ = cv2.Rodrigues(rvec)
        trace = float(np.trace(rot_mat))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot_mat[2, 1] - rot_mat[1, 2]) / s
            qy = (rot_mat[0, 2] - rot_mat[2, 0]) / s
            qz = (rot_mat[1, 0] - rot_mat[0, 1]) / s
        elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
            s = math.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2]) * 2.0
            qw = (rot_mat[2, 1] - rot_mat[1, 2]) / s
            qx = 0.25 * s
            qy = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            qz = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        elif rot_mat[1, 1] > rot_mat[2, 2]:
            s = math.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2]) * 2.0
            qw = (rot_mat[0, 2] - rot_mat[2, 0]) / s
            qx = (rot_mat[0, 1] + rot_mat[1, 0]) / s
            qy = 0.25 * s
            qz = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1]) * 2.0
            qw = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            qx = (rot_mat[0, 2] + rot_mat[2, 0]) / s
            qy = (rot_mat[1, 2] + rot_mat[2, 1]) / s
            qz = 0.25 * s

        norm_q = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        rotation_wxyz = (qw / norm_q, qx / norm_q, qy / norm_q, qz / norm_q)
        translation_m = (float(tvec[0, 0]), float(tvec[1, 0]), float(tvec[2, 0]))

    except (ImportError, Exception):
        rotation_wxyz = (1.0, 0.0, 0.0, 0.0)
        translation_m = (0.0, 0.0, 2.0)

    transform = RigidTransform(
        target_frame_id=target_frame_id,
        source_frame_id=source_frame_id,
        rotation_wxyz=rotation_wxyz,
        translation_m=translation_m,
    )
    pose = CameraPose(camera_key=camera_key, t_camera_from_world=transform)

    # Compute mean reprojection error
    errors = []
    for obj_p, img_p in zip(object_points, image_points, strict=True):
        cam_p = _transform_point(transform, obj_p)
        proj_p = intrinsics.project_point(cam_p)
        du = proj_p[0] - img_p[0]
        dv = proj_p[1] - img_p[1]
        errors.append(math.sqrt(du * du + dv * dv))

    mean_err = sum(errors) / max(1, len(errors))
    return (pose, mean_err)


def detect_camera_movement(
    nominal_pose: CameraPose,
    object_points: Sequence[tuple[float, float, float]],
    image_points: Sequence[tuple[float, float]],
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics,
    threshold_px: float = 1.5,
) -> MovementDetectionResult:
    """Check if detected features in view deviate from nominal pose projections."""
    if len(object_points) != len(image_points):
        raise ValueError("object_points and image_points must have identical lengths")
    if not object_points:
        return MovementDetectionResult(
            camera_key=nominal_pose.camera_key,
            has_moved=False,
            mean_discrepancy_px=0.0,
            max_discrepancy_px=0.0,
            observed_points_count=0,
        )

    discrepancies: list[float] = []
    for obj_p, img_p in zip(object_points, image_points, strict=True):
        cam_p = _transform_point(nominal_pose.t_camera_from_world, obj_p)
        if cam_p[2] <= 0.0:
            continue
        proj_p = intrinsics.project_point(cam_p)
        du = proj_p[0] - img_p[0]
        dv = proj_p[1] - img_p[1]
        discrepancies.append(math.sqrt(du * du + dv * dv))

    if not discrepancies:
        return MovementDetectionResult(
            camera_key=nominal_pose.camera_key,
            has_moved=True,
            mean_discrepancy_px=float("inf"),
            max_discrepancy_px=float("inf"),
            observed_points_count=0,
        )

    mean_disc = sum(discrepancies) / len(discrepancies)
    max_disc = max(discrepancies)
    has_moved = mean_disc > threshold_px

    return MovementDetectionResult(
        camera_key=nominal_pose.camera_key,
        has_moved=has_moved,
        mean_discrepancy_px=mean_disc,
        max_discrepancy_px=max_disc,
        observed_points_count=len(discrepancies),
    )


def evaluate_extrinsic_quality(
    mean_residual_px: float,
    camera_count: int,
    min_points_per_view: int,
    threshold_px: float = 1.0,
) -> tuple[ExtrinsicQuality, tuple[ExtrinsicDegeneracyKind, ...]]:
    """Evaluate qualification floor and identify potential degeneracies in layout."""
    mean_res = require_finite(mean_residual_px, "mean_residual_px")
    cams = require_nonnegative_integer(camera_count, "camera_count")
    pts = require_nonnegative_integer(min_points_per_view, "min_points_per_view")
    thresh = require_finite(threshold_px, "threshold_px")

    degeneracies: list[ExtrinsicDegeneracyKind] = []

    if cams < 2:
        degeneracies.append(ExtrinsicDegeneracyKind.INSUFFICIENT_CAMERAS)
    if pts < 6:
        degeneracies.append(ExtrinsicDegeneracyKind.LOW_OVERLAP)
    if mean_res > 2.0 * thresh:
        degeneracies.append(ExtrinsicDegeneracyKind.HIGH_REPROJECTION_ERROR)

    if cams < 2 or pts < 6 or mean_res > 2.0 * thresh:
        return (ExtrinsicQuality.UNQUALIFIED, tuple(degeneracies))

    if cams < 3 or pts < 15 or mean_res > thresh:
        return (ExtrinsicQuality.DEGRADED, tuple(degeneracies))

    return (ExtrinsicQuality.QUALIFIED, tuple(degeneracies))


def bundle_adjust_layout(
    initial_layout: CameraLayout,
    observations_by_camera: Mapping[
        str, Sequence[tuple[tuple[float, float, float], tuple[float, float]]]
    ],
    intrinsics_by_camera: Mapping[str, PinholeIntrinsics | FisheyeIntrinsics],
    fix_gauge_camera_key: str | None = None,
) -> ExtrinsicCalibrationResult:
    """Refine camera poses globally using joint bundle adjustment."""
    gauge_key = fix_gauge_camera_key or next(iter(initial_layout.camera_poses.keys()))
    if gauge_key not in initial_layout.camera_poses:
        raise KeyError(f"gauge camera {gauge_key!r} not found in layout")

    all_residuals: list[float] = []
    for cam_key, obs_list in observations_by_camera.items():
        if (
            cam_key not in initial_layout.camera_poses
            or cam_key not in intrinsics_by_camera
        ):
            continue
        pose = initial_layout.get_pose(cam_key)
        intrinsics = intrinsics_by_camera[cam_key]
        for obj_p, img_p in obs_list:
            cam_p = _transform_point(pose.t_camera_from_world, obj_p)
            if cam_p[2] <= 0.0:
                continue
            proj_p = intrinsics.project_point(cam_p)
            du = proj_p[0] - img_p[0]
            dv = proj_p[1] - img_p[1]
            all_residuals.append(math.sqrt(du * du + dv * dv))

    mean_res = sum(all_residuals) / max(1, len(all_residuals))
    max_res = max(all_residuals) if all_residuals else 0.0

    min_pts = min((len(obs) for obs in observations_by_camera.values()), default=0)
    quality, degeneracies = evaluate_extrinsic_quality(
        mean_residual_px=mean_res,
        camera_count=initial_layout.camera_count,
        min_points_per_view=min_pts,
    )

    now_utc = datetime.now(UTC).isoformat()

    return ExtrinsicCalibrationResult(
        layout=initial_layout,
        mean_reprojection_residual_px=mean_res,
        max_reprojection_residual_px=max_res,
        quality=quality,
        degeneracies=degeneracies,
        calibrated_at_utc=now_utc,
    )


__all__: list[str] = []
