"""Contract tests for multi-view association and triangulation
(TOOLS-M7 #4724).
"""

from __future__ import annotations

import math

import pytest
from sidekick.lab.mocap.calibration import (
    DistortionCoefficients,
    DistortionModel,
    PinholeIntrinsics,
)
from sidekick.lab.mocap.enums import Availability
from sidekick.lab.mocap.extrinsics import CameraLayout, CameraPose
from sidekick.lab.mocap.geometry import CoordinateFrame, RigidTransform
from sidekick.lab.mocap.observations import PixelObservation
from sidekick.lab.mocap.reconstruction import (
    KeypointReconstruction,
    ReconstructionConfig,
    ReconstructionQuality,
    reconstruct_frame_landmarks,
    triangulate_n_views,
)


def _make_intrinsics(
    w: int = 1920, h: int = 1080, f: float = 1200.0
) -> PinholeIntrinsics:
    return PinholeIntrinsics(
        fx=f,
        fy=f,
        cx=w / 2.0,
        cy=h / 2.0,
        resolution_px=(w, h),
        distortion=DistortionCoefficients(DistortionModel.NONE, ()),
    )


def _rotation_matrix_to_quaternion(
    r: list[list[float]],
) -> tuple[float, float, float, float]:
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return (
            0.25 * s,
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
        )
    elif (r[0][0] > r[1][1]) and (r[0][0] > r[2][2]):
        s = math.sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2.0
        return (
            (r[2][1] - r[1][2]) / s,
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
        )
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2.0
        return (
            (r[0][2] - r[2][0]) / s,
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
        )
    else:
        s = math.sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2.0
        return (
            (r[1][0] - r[0][1]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
        )


def _look_at_pose(
    camera_key: str,
    cam_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float] = (0.0, 1.0, 0.0),
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> CameraPose:
    # Camera looks in +Z direction in OpenCV convention, X right, Y down
    # Forward vector z = target - cam_pos
    zx = target_pos[0] - cam_pos[0]
    zy = target_pos[1] - cam_pos[1]
    zz = target_pos[2] - cam_pos[2]
    norm_z = math.sqrt(zx * zx + zy * zy + zz * zz)
    zx, zy, zz = zx / norm_z, zy / norm_z, zz / norm_z

    # Right vector x = up x z
    xx = up[1] * zz - up[2] * zy
    xy = up[2] * zx - up[0] * zz
    xz = up[0] * zy - up[1] * zx
    norm_x = math.sqrt(xx * xx + xy * xy + xz * xz)
    xx, xy, xz = xx / norm_x, xy / norm_x, xz / norm_x

    # Down vector y = z x x
    yx = zy * xz - zz * xy
    yy = zz * xx - zx * xz
    yz = zx * xy - zy * xx

    # Rotation matrix R maps world to camera
    R = [
        [xx, xy, xz],
        [yx, yy, yz],
        [zx, zy, zz],
    ]
    # translation t = -R * cam_pos
    tx = -(xx * cam_pos[0] + xy * cam_pos[1] + xz * cam_pos[2])
    ty = -(yx * cam_pos[0] + yy * cam_pos[1] + yz * cam_pos[2])
    tz = -(zx * cam_pos[0] + zy * cam_pos[1] + zz * cam_pos[2])

    quat = _rotation_matrix_to_quaternion(R)
    transform = RigidTransform(
        target_frame_id=f"frame-{camera_key}",
        source_frame_id="affinedrift-world-v1",
        rotation_wxyz=quat,
        translation_m=(tx, ty, tz),
    )
    return CameraPose(camera_key=camera_key, t_camera_from_world=transform)


def _setup_4_camera_rig() -> tuple[CameraLayout, dict[str, PinholeIntrinsics]]:
    world_frame = CoordinateFrame.affinedrift_world_v1()
    cams = {
        "cam-01": _look_at_pose("cam-01", (2.5, 1.5, 2.5)),
        "cam-02": _look_at_pose("cam-02", (-2.5, 1.5, 2.5)),
        "cam-03": _look_at_pose("cam-03", (-2.5, 1.5, -2.5)),
        "cam-04": _look_at_pose("cam-04", (2.5, 1.5, -2.5)),
    }
    layout = CameraLayout(
        layout_id="rig-4cam-v1", world_frame=world_frame, camera_poses=cams
    )
    intrinsics = {k: _make_intrinsics() for k in cams}
    return layout, intrinsics


def _project_world_point(
    point_world: tuple[float, float, float],
    pose: CameraPose,
    intrinsics: PinholeIntrinsics,
) -> tuple[float, float]:
    qw, qx, qy, qz = pose.t_camera_from_world.rotation_wxyz
    tx, ty, tz = pose.t_camera_from_world.translation_m

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qz * qw)
    r02 = 2.0 * (qx * qz + qy * qw)
    r10 = 2.0 * (qx * qy + qz * qw)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qx * qw)
    r20 = 2.0 * (qx * qz - qy * qw)
    r21 = 2.0 * (qy * qz + qx * qw)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    wx, wy, wz = point_world
    cx = r00 * wx + r01 * wy + r02 * wz + tx
    cy = r10 * wx + r11 * wy + r12 * wz + ty
    cz = r20 * wx + r21 * wy + r22 * wz + tz

    proj = intrinsics.project_point((cx, cy, cz))
    return (float(proj[0]), float(proj[1]))


def test_triangulate_n_views_exact_recovery() -> None:
    layout, intrinsics = _setup_4_camera_rig()
    pt_true = (0.2, 1.1, -0.1)

    obs = []
    for cam_id in ("cam-01", "cam-02", "cam-03", "cam-04"):
        uv = _project_world_point(pt_true, layout.get_pose(cam_id), intrinsics[cam_id])
        obs.append(
            PixelObservation(
                observation_id=f"obs-{cam_id}",
                camera_id=cam_id,
                frame_sequence=100,
                timestamp_ns=1_000_000_000,
                skeleton_id="mediapipe-pose-33-v1",
                keypoint_id="nose",
                uv_px=uv,
                confidence=0.95,
                covariance_px2=(1.0, 0.0, 0.0, 1.0),
                availability=Availability.OBSERVED,
            )
        )

    result = triangulate_n_views(
        observations=obs,
        camera_layout=layout,
        camera_intrinsics=intrinsics,
        config=ReconstructionConfig(),
    )

    assert isinstance(result, KeypointReconstruction)
    assert result.quality is ReconstructionQuality.QUALIFIED
    assert len(result.inlier_camera_ids) == 4
    assert len(result.outlier_camera_ids) == 0

    x, y, z = result.landmark.xyz_m
    assert pytest.approx(pt_true[0], abs=1e-3) == x
    assert pytest.approx(pt_true[1], abs=1e-3) == y
    assert pytest.approx(pt_true[2], abs=1e-3) == z
    assert result.landmark.availability is Availability.DERIVED


def test_triangulate_n_views_outlier_rejection() -> None:
    layout, intrinsics = _setup_4_camera_rig()
    pt_true = (0.0, 1.0, 0.0)

    obs = []
    for cam_id in ("cam-01", "cam-02", "cam-03"):
        uv = _project_world_point(pt_true, layout.get_pose(cam_id), intrinsics[cam_id])
        obs.append(
            PixelObservation(
                observation_id=f"obs-{cam_id}",
                camera_id=cam_id,
                frame_sequence=100,
                timestamp_ns=1_000_000_000,
                skeleton_id="mediapipe-pose-33-v1",
                keypoint_id="left_wrist",
                uv_px=uv,
                confidence=0.9,
                covariance_px2=(1.0, 0.0, 0.0, 1.0),
                availability=Availability.OBSERVED,
            )
        )

    # Corrupted 4th camera observation (gross outlier shifted by 150px)
    uv_corrupt = _project_world_point(
        pt_true, layout.get_pose("cam-04"), intrinsics["cam-04"]
    )
    obs.append(
        PixelObservation(
            observation_id="obs-cam-04",
            camera_id="cam-04",
            frame_sequence=100,
            timestamp_ns=1_000_000_000,
            skeleton_id="mediapipe-pose-33-v1",
            keypoint_id="left_wrist",
            uv_px=(uv_corrupt[0] + 150.0, uv_corrupt[1] + 150.0),
            confidence=0.9,
            covariance_px2=(1.0, 0.0, 0.0, 1.0),
            availability=Availability.OBSERVED,
        )
    )

    result = triangulate_n_views(
        observations=obs,
        camera_layout=layout,
        camera_intrinsics=intrinsics,
        config=ReconstructionConfig(max_reprojection_error_px=5.0),
    )

    assert "cam-04" in result.outlier_camera_ids
    assert "cam-04" in result.landmark.rejected_camera_ids
    assert set(result.inlier_camera_ids) == {"cam-01", "cam-02", "cam-03"}
    assert set(result.landmark.contributing_camera_ids) == {
        "cam-01",
        "cam-02",
        "cam-03",
    }
    assert result.quality is ReconstructionQuality.QUALIFIED

    x, y, z = result.landmark.xyz_m
    assert pytest.approx(pt_true[0], abs=1e-2) == x
    assert pytest.approx(pt_true[1], abs=1e-2) == y
    assert pytest.approx(pt_true[2], abs=1e-2) == z


def test_triangulate_n_views_minimum_views_fail_closed() -> None:
    layout, intrinsics = _setup_4_camera_rig()
    pt_true = (0.0, 1.0, 0.0)

    # Only 1 observation provided -> cannot triangulate 3D point
    uv = _project_world_point(pt_true, layout.get_pose("cam-01"), intrinsics["cam-01"])
    obs = [
        PixelObservation(
            observation_id="obs-cam-01",
            camera_id="cam-01",
            frame_sequence=100,
            timestamp_ns=1_000_000_000,
            skeleton_id="mediapipe-pose-33-v1",
            keypoint_id="nose",
            uv_px=uv,
            confidence=0.95,
            covariance_px2=(1.0, 0.0, 0.0, 1.0),
            availability=Availability.OBSERVED,
        )
    ]

    result = triangulate_n_views(
        observations=obs,
        camera_layout=layout,
        camera_intrinsics=intrinsics,
        config=ReconstructionConfig(min_cameras=2),
    )

    assert result.quality is ReconstructionQuality.UNQUALIFIED
    assert result.landmark.availability is Availability.UNAVAILABLE
    assert len(result.inlier_camera_ids) == 0


def test_reconstruct_frame_landmarks_multiple_keypoints() -> None:
    layout, intrinsics = _setup_4_camera_rig()
    points = {
        "nose": (0.0, 1.7, 0.0),
        "left_shoulder": (0.2, 1.4, 0.0),
        "right_shoulder": (-0.2, 1.4, 0.0),
    }

    obs = []
    for kp, pt in points.items():
        for cam_id in ("cam-01", "cam-02", "cam-03"):
            uv = _project_world_point(pt, layout.get_pose(cam_id), intrinsics[cam_id])
            obs.append(
                PixelObservation(
                    observation_id=f"obs-{cam_id}-{kp}",
                    camera_id=cam_id,
                    frame_sequence=10,
                    timestamp_ns=100_000_000,
                    skeleton_id="mediapipe-pose-33-v1",
                    keypoint_id=kp,
                    uv_px=uv,
                    confidence=0.9,
                    covariance_px2=(1.0, 0.0, 0.0, 1.0),
                    availability=Availability.OBSERVED,
                )
            )

    results = reconstruct_frame_landmarks(
        observations=obs,
        camera_layout=layout,
        camera_intrinsics=intrinsics,
        config=ReconstructionConfig(),
    )

    assert len(results) == 3
    reconstructed_kps = {r.landmark.keypoint_id: r for r in results}
    assert set(reconstructed_kps.keys()) == {"nose", "left_shoulder", "right_shoulder"}
    for kp, pt in points.items():
        r = reconstructed_kps[kp]
        assert r.quality is ReconstructionQuality.QUALIFIED
        assert pytest.approx(pt[0], abs=1e-2) == r.landmark.xyz_m[0]
        assert pytest.approx(pt[1], abs=1e-2) == r.landmark.xyz_m[1]
        assert pytest.approx(pt[2], abs=1e-2) == r.landmark.xyz_m[2]
