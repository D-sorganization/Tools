"""Cross-view observation association, robust multi-view triangulation,
and 3-D landmark reconstruction.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from ._validation import require_finite
from .calibration import FisheyeIntrinsics, PinholeIntrinsics
from .enums import Availability
from .extrinsics import CameraLayout, CameraPose
from .observations import Landmark3D, PixelObservation


class ReconstructionQuality(StrEnum):
    """Categorical qualification floor for multi-view landmark reconstruction."""

    QUALIFIED = "qualified"
    DEGRADED = "degraded"
    UNQUALIFIED = "unqualified"


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    """Parameters for association, triangulation, and outlier rejection."""

    min_cameras: int = 2
    max_reprojection_error_px: float = 8.0
    min_confidence: float = 0.2
    outlier_rejection_enabled: bool = True

    def __post_init__(self) -> None:
        if self.min_cameras < 2:
            raise ValueError("min_cameras must be at least 2")
        require_finite(self.max_reprojection_error_px, "max_reprojection_error_px")
        if self.max_reprojection_error_px <= 0.0:
            raise ValueError("max_reprojection_error_px must be positive")
        require_finite(self.min_confidence, "min_confidence")
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class KeypointReconstruction:
    """Outcome of 3-D triangulation for a single keypoint across multi-camera views."""

    landmark: Landmark3D
    quality: ReconstructionQuality
    reprojection_errors_px: Mapping[str, float]
    inlier_camera_ids: tuple[str, ...]
    outlier_camera_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.landmark, Landmark3D):
            raise TypeError("landmark must be a Landmark3D")
        if not isinstance(self.quality, ReconstructionQuality):
            raise TypeError("quality must be a ReconstructionQuality")
        object.__setattr__(
            self, "reprojection_errors_px", dict(self.reprojection_errors_px)
        )


def _rotation_matrix_from_pose(
    pose: CameraPose,
) -> tuple[tuple[float, float, float], ...]:
    """Compute 3x3 rotation matrix from CameraPose quaternion."""
    qw, qx, qy, qz = pose.t_camera_from_world.rotation_wxyz
    return (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )


def _camera_world_projection_matrix(
    pose: CameraPose,
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics,
) -> list[list[float]]:
    """Compute 3x4 camera projection matrix P = K * [R | t] in world frame."""
    r = _rotation_matrix_from_pose(pose)
    tx, ty, tz = pose.t_camera_from_world.translation_m
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.cx, intrinsics.cy
    skew = getattr(intrinsics, "skew", 0.0)

    return [
        [
            fx * r[0][0] + skew * r[1][0] + cx * r[2][0],
            fx * r[0][1] + skew * r[1][1] + cx * r[2][1],
            fx * r[0][2] + skew * r[1][2] + cx * r[2][2],
            fx * tx + skew * ty + cx * tz,
        ],
        [
            fy * r[1][0] + cy * r[2][0],
            fy * r[1][1] + cy * r[2][1],
            fy * r[1][2] + cy * r[2][2],
            fy * ty + cy * tz,
        ],
        [r[2][0], r[2][1], r[2][2], tz],
    ]


def _invert_3x3(m: Sequence[Sequence[float]]) -> list[list[float]] | None:
    """Invert 3x3 matrix using analytic cofactors, returning None if singular."""
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    return [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]


def _solve_dlt_point(
    projections: Sequence[tuple[list[list[float]], tuple[float, float], float]],
) -> tuple[float, float, float]:
    """Solve Direct Linear Transform (DLT) weighted linear least squares."""
    ata = [[0.0, 0.0, 0.0] for _ in range(3)]
    atb = [0.0, 0.0, 0.0]

    for p, (u, v), weight in projections:
        w2 = weight * weight
        r1 = (u * p[2][0] - p[0][0], u * p[2][1] - p[0][1], u * p[2][2] - p[0][2])
        d1 = p[0][3] - u * p[2][3]
        r2 = (v * p[2][0] - p[1][0], v * p[2][1] - p[1][1], v * p[2][2] - p[1][2])
        d2 = p[1][3] - v * p[2][3]

        for (a, b, c), d in ((r1, d1), (r2, d2)):
            row = (a, b, c)
            for i in range(3):
                atb[i] += w2 * row[i] * d
                for j in range(3):
                    ata[i][j] += w2 * row[i] * row[j]

    inv = _invert_3x3(ata)
    if inv is None:
        raise ValueError("singular system in DLT triangulation")

    x = inv[0][0] * atb[0] + inv[0][1] * atb[1] + inv[0][2] * atb[2]
    y = inv[1][0] * atb[0] + inv[1][1] * atb[1] + inv[1][2] * atb[2]
    z = inv[2][0] * atb[0] + inv[2][1] * atb[1] + inv[2][2] * atb[2]
    return (x, y, z)


def _compute_reprojection_error(
    point_world: tuple[float, float, float],
    pose: CameraPose,
    intrinsics: PinholeIntrinsics | FisheyeIntrinsics,
    observed_uv: tuple[float, float],
) -> float:
    """Compute Euclidean pixel reprojection error against 2D observation."""
    r = _rotation_matrix_from_pose(pose)
    tx, ty, tz = pose.t_camera_from_world.translation_m

    wx, wy, wz = point_world
    cx = r[0][0] * wx + r[0][1] * wy + r[0][2] * wz + tx
    cy = r[1][0] * wx + r[1][1] * wy + r[1][2] * wz + ty
    cz = r[2][0] * wx + r[2][1] * wy + r[2][2] * wz + tz

    if cz <= 0.0:
        return 1e6

    proj_uv = intrinsics.project_point((cx, cy, cz))
    du = proj_uv[0] - observed_uv[0]
    dv = proj_uv[1] - observed_uv[1]
    return math.sqrt(du * du + dv * dv)


def _estimate_spatial_covariance(
    point_world: tuple[float, float, float],
    inliers: Sequence[
        tuple[CameraPose, PinholeIntrinsics | FisheyeIntrinsics, PixelObservation]
    ],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Compute 3x3 spatial covariance matrix (J^T * W * J)^-1 in world coordinates."""
    jtwj = [[0.0, 0.0, 0.0] for _ in range(3)]
    for pose, intrinsics, obs in inliers:
        r = _rotation_matrix_from_pose(pose)
        tx, ty, tz = pose.t_camera_from_world.translation_m

        wx, wy, wz = point_world
        cx = r[0][0] * wx + r[0][1] * wy + r[0][2] * wz + tx
        cy = r[1][0] * wx + r[1][1] * wy + r[1][2] * wz + ty
        cz = r[2][0] * wx + r[2][1] * wy + r[2][2] * wz + tz
        if cz <= 0.0:
            continue

        fx, fy = intrinsics.fx, intrinsics.fy
        inv_cz = 1.0 / cz
        inv_cz2 = inv_cz * inv_cz
        du_dc = (fx * inv_cz, 0.0, -fx * cx * inv_cz2)
        dv_dc = (0.0, fy * inv_cz, -fy * cy * inv_cz2)

        j_u = [
            du_dc[0] * r[0][0] + du_dc[1] * r[1][0] + du_dc[2] * r[2][0],
            du_dc[0] * r[0][1] + du_dc[1] * r[1][1] + du_dc[2] * r[2][1],
            du_dc[0] * r[0][2] + du_dc[1] * r[1][2] + du_dc[2] * r[2][2],
        ]
        j_v = [
            dv_dc[0] * r[0][0] + dv_dc[1] * r[1][0] + dv_dc[2] * r[2][0],
            dv_dc[0] * r[0][1] + dv_dc[1] * r[1][1] + dv_dc[2] * r[2][1],
            dv_dc[0] * r[0][2] + dv_dc[1] * r[1][2] + dv_dc[2] * r[2][2],
        ]

        c_u = max(1e-4, obs.covariance_px2[0])
        c_v = max(1e-4, obs.covariance_px2[3])
        w_u = obs.confidence / c_u
        w_v = obs.confidence / c_v

        for i in range(3):
            for j in range(3):
                jtwj[i][j] += w_u * j_u[i] * j_u[j] + w_v * j_v[i] * j_v[j]

    inv = _invert_3x3(jtwj)
    if inv is None:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    return (
        inv[0][0],
        inv[0][1],
        inv[0][2],
        inv[1][0],
        inv[1][1],
        inv[1][2],
        inv[2][0],
        inv[2][1],
        inv[2][2],
    )


def triangulate_n_views(
    observations: Sequence[PixelObservation],
    camera_layout: CameraLayout,
    camera_intrinsics: Mapping[str, PinholeIntrinsics | FisheyeIntrinsics],
    config: ReconstructionConfig | None = None,
) -> KeypointReconstruction:
    """Perform robust, confidence-weighted N-view triangulation."""
    cfg = config or ReconstructionConfig()
    if not observations:
        raise ValueError("observations must not be empty")

    sample = observations[0]
    skeleton_id = sample.skeleton_id
    keypoint_id = sample.keypoint_id
    timestamp_ns = sample.timestamp_ns
    frame_sequence = sample.frame_sequence
    world_frame_id = camera_layout.world_frame.frame_id

    valid_obs: list[
        tuple[CameraPose, PinholeIntrinsics | FisheyeIntrinsics, PixelObservation]
    ] = []
    rejected_cams: list[str] = []

    for obs in observations:
        if (
            obs.availability is not Availability.OBSERVED
            or obs.confidence < cfg.min_confidence
        ):
            rejected_cams.append(obs.camera_id)
            continue
        if (
            obs.camera_id not in camera_layout.camera_poses
            or obs.camera_id not in camera_intrinsics
        ):
            rejected_cams.append(obs.camera_id)
            continue
        pose = camera_layout.get_pose(obs.camera_id)
        intrinsics = camera_intrinsics[obs.camera_id]
        valid_obs.append((pose, intrinsics, obs))

    if len(valid_obs) < cfg.min_cameras:
        landmark = Landmark3D(
            landmark_id=f"lmk-{skeleton_id}-{keypoint_id}-{frame_sequence}",
            world_frame_id=world_frame_id,
            skeleton_id=skeleton_id,
            keypoint_id=keypoint_id,
            timestamp_ns=timestamp_ns,
            xyz_m=(0.0, 0.0, 0.0),
            covariance_m2=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            contributing_camera_ids=(),
            rejected_camera_ids=tuple(
                sorted(set(rejected_cams + [o.camera_id for _, _, o in valid_obs]))
            ),
            method_id="mocap-triangulation-dlt-v1",
            availability=Availability.UNAVAILABLE,
        )
        return KeypointReconstruction(
            landmark=landmark,
            quality=ReconstructionQuality.UNQUALIFIED,
            reprojection_errors_px={},
            inlier_camera_ids=(),
            outlier_camera_ids=landmark.rejected_camera_ids,
        )

    projections = [
        (
            _camera_world_projection_matrix(pose, intrinsics),
            obs.uv_px,
            obs.confidence,
        )
        for pose, intrinsics, obs in valid_obs
    ]
    try:
        xyz_all = _solve_dlt_point(projections)
    except ValueError:
        xyz_all = (0.0, 0.0, 0.0)

    errors_all = {
        obs.camera_id: _compute_reprojection_error(xyz_all, pose, intrinsics, obs.uv_px)
        for pose, intrinsics, obs in valid_obs
    }
    inliers_all = [
        item
        for item in valid_obs
        if errors_all[item[2].camera_id] <= cfg.max_reprojection_error_px
    ]

    best_inliers = inliers_all
    best_xyz = xyz_all
    best_errors = errors_all

    # If outlier rejection enabled and some views violated threshold,
    # search subsets for maximum consensus
    if cfg.outlier_rejection_enabled and (
        len(inliers_all) < len(valid_obs) or len(inliers_all) < cfg.min_cameras
    ):
        max_inlier_count = len(inliers_all)
        for k in range(cfg.min_cameras, len(valid_obs)):
            for subset in itertools.combinations(valid_obs, k):
                try:
                    sub_proj = [
                        (
                            _camera_world_projection_matrix(pose, intrinsics),
                            obs.uv_px,
                            obs.confidence,
                        )
                        for pose, intrinsics, obs in subset
                    ]
                    sub_xyz = _solve_dlt_point(sub_proj)
                except ValueError:
                    continue

                sub_errors = {
                    obs.camera_id: _compute_reprojection_error(
                        sub_xyz, pose, intrinsics, obs.uv_px
                    )
                    for pose, intrinsics, obs in valid_obs
                }
                sub_inliers = [
                    item
                    for item in valid_obs
                    if sub_errors[item[2].camera_id] <= cfg.max_reprojection_error_px
                ]
                if len(sub_inliers) > max_inlier_count:
                    max_inlier_count = len(sub_inliers)
                    best_inliers = sub_inliers
                    best_xyz = sub_xyz
                    best_errors = sub_errors

    if len(best_inliers) >= cfg.min_cameras:
        if len(best_inliers) < len(valid_obs):
            inlier_proj = [
                (
                    _camera_world_projection_matrix(pose, intrinsics),
                    obs.uv_px,
                    obs.confidence,
                )
                for pose, intrinsics, obs in best_inliers
            ]
            try:
                best_xyz = _solve_dlt_point(inlier_proj)
            except ValueError:
                pass
            best_errors = {
                obs.camera_id: _compute_reprojection_error(
                    best_xyz, pose, intrinsics, obs.uv_px
                )
                for pose, intrinsics, obs in valid_obs
            }

        inlier_cams = set(item[2].camera_id for item in best_inliers)
        outlier_cams = [
            item[2].camera_id
            for item in valid_obs
            if item[2].camera_id not in inlier_cams
        ]

        mean_inlier_err = sum(best_errors[c] for c in inlier_cams) / max(
            1, len(inlier_cams)
        )
        quality = (
            ReconstructionQuality.QUALIFIED
            if len(best_inliers) >= 3
            and mean_inlier_err <= cfg.max_reprojection_error_px
            else ReconstructionQuality.DEGRADED
            if len(best_inliers) >= cfg.min_cameras
            else ReconstructionQuality.UNQUALIFIED
        )
        avail = Availability.DERIVED
        inliers_tuple = tuple(sorted(inlier_cams))
        outliers_tuple = tuple(sorted(set(rejected_cams + outlier_cams)))
        cov = _estimate_spatial_covariance(best_xyz, best_inliers)
    else:
        quality = ReconstructionQuality.UNQUALIFIED
        avail = Availability.UNAVAILABLE
        inliers_tuple = ()
        outliers_tuple = tuple(
            sorted(set(rejected_cams + [o.camera_id for _, _, o in valid_obs]))
        )
        cov = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        best_xyz = (0.0, 0.0, 0.0)

    landmark = Landmark3D(
        landmark_id=f"lmk-{skeleton_id}-{keypoint_id}-{frame_sequence}",
        world_frame_id=world_frame_id,
        skeleton_id=skeleton_id,
        keypoint_id=keypoint_id,
        timestamp_ns=timestamp_ns,
        xyz_m=best_xyz if avail is Availability.DERIVED else (0.0, 0.0, 0.0),
        covariance_m2=cov,
        contributing_camera_ids=inliers_tuple,
        rejected_camera_ids=outliers_tuple,
        method_id="mocap-triangulation-dlt-v1",
        availability=avail,
    )

    return KeypointReconstruction(
        landmark=landmark,
        quality=quality,
        reprojection_errors_px=best_errors,
        inlier_camera_ids=inliers_tuple,
        outlier_camera_ids=outliers_tuple,
    )


def reconstruct_frame_landmarks(
    observations: Sequence[PixelObservation],
    camera_layout: CameraLayout,
    camera_intrinsics: Mapping[str, PinholeIntrinsics | FisheyeIntrinsics],
    config: ReconstructionConfig | None = None,
) -> tuple[KeypointReconstruction, ...]:
    """Reconstruct all 3-D landmarks for a frame across skeleton keypoints."""
    groups: dict[tuple[str, str], list[PixelObservation]] = {}
    for obs in observations:
        key = (obs.skeleton_id, obs.keypoint_id)
        if key not in groups:
            groups[key] = []
        groups[key].append(obs)

    results: list[KeypointReconstruction] = []
    for _key, group in groups.items():
        rec = triangulate_n_views(
            observations=group,
            camera_layout=camera_layout,
            camera_intrinsics=camera_intrinsics,
            config=config,
        )
        results.append(rec)

    return tuple(results)
