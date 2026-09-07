"""TDD contract tests for markerless-mocap extrinsic calibration (#4721)."""

from __future__ import annotations

import math

import pytest
from sidekick.lab.mocap.calibration import (
    DistortionCoefficients,
    DistortionModel,
    PinholeIntrinsics,
)
from sidekick.lab.mocap.extrinsics import (
    CameraLayout,
    CameraPose,
    ExtrinsicDegeneracyKind,
    ExtrinsicQuality,
    bundle_adjust_layout,
    detect_camera_movement,
    estimate_pnp_pose,
    evaluate_extrinsic_quality,
)
from sidekick.lab.mocap.geometry import CoordinateFrame, RigidTransform


def _sample_pinhole() -> PinholeIntrinsics:
    return PinholeIntrinsics(
        fx=1000.0,
        fy=1000.0,
        cx=960.0,
        cy=540.0,
        resolution_px=(1920, 1080),
        distortion=DistortionCoefficients(model=DistortionModel.NONE, coefficients=()),
    )


def test_camera_pose_contracts() -> None:
    # Camera pose transform maps world -> camera
    t_cam_from_world = RigidTransform(
        target_frame_id="cam_0",
        source_frame_id="world",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_m=(0.0, 0.0, -2.0),
    )
    pose = CameraPose(
        camera_key="cam_0",
        t_camera_from_world=t_cam_from_world,
    )
    assert pose.camera_key == "cam_0"
    assert pose.t_camera_from_world == t_cam_from_world
    # Center of camera in world frame: R^T * (-t)
    assert pose.camera_center_world == (0.0, 0.0, 2.0)


def test_camera_layout_contracts() -> None:
    world_frame = CoordinateFrame.affinedrift_world_v1()
    t0 = RigidTransform(
        target_frame_id="cam_0",
        source_frame_id="world",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_m=(0.0, 0.0, -2.0),
    )
    t1 = RigidTransform(
        target_frame_id="cam_1",
        source_frame_id="world",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_m=(-1.0, 0.0, -2.0),
    )
    pose0 = CameraPose(camera_key="cam_0", t_camera_from_world=t0)
    pose1 = CameraPose(camera_key="cam_1", t_camera_from_world=t1)

    layout = CameraLayout(
        layout_id="layout_2cam_v1",
        world_frame=world_frame,
        camera_poses={"cam_0": pose0, "cam_1": pose1},
    )
    assert layout.camera_count == 2
    assert "cam_0" in layout.camera_poses
    assert layout.get_pose("cam_0") == pose0

    # Layout requires at least 1 camera
    with pytest.raises(ValueError, match="layout must contain at least one camera"):
        CameraLayout(
            layout_id="empty_layout",
            world_frame=world_frame,
            camera_poses={},
        )


def test_estimate_pnp_pose_reproducible() -> None:
    intrinsics = _sample_pinhole()
    obj_pts = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.2, 0.2, 0.0),
        (0.0, 0.2, 0.0),
        (0.1, 0.1, 0.0),
        (0.05, 0.15, 0.0),
    )
    img_pts = tuple(intrinsics.project_point((x, y, z + 2.0)) for x, y, z in obj_pts)

    pose, residual = estimate_pnp_pose(
        camera_key="cam_0",
        object_points=obj_pts,
        image_points=img_pts,
        intrinsics=intrinsics,
        target_frame_id="cam_0",
        source_frame_id="world",
    )
    assert pose.camera_key == "cam_0"
    assert residual < 1e-3
    assert math.isclose(pose.t_camera_from_world.translation_m[2], 2.0, abs_tol=1e-3)


def test_movement_invalidation_and_relocalization() -> None:
    intrinsics = _sample_pinhole()
    t_nominal = RigidTransform(
        target_frame_id="cam_0",
        source_frame_id="world",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_m=(0.0, 0.0, 2.0),
    )
    nominal_pose = CameraPose(camera_key="cam_0", t_camera_from_world=t_nominal)

    obj_pts = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.2, 0.2, 0.0),
        (0.0, 0.2, 0.0),
        (0.1, 0.1, 0.0),
    )
    img_pts_undisturbed = tuple(
        intrinsics.project_point((x, y, z + 2.0)) for x, y, z in obj_pts
    )
    mov_check = detect_camera_movement(
        nominal_pose=nominal_pose,
        object_points=obj_pts,
        image_points=img_pts_undisturbed,
        intrinsics=intrinsics,
        threshold_px=1.5,
    )
    assert not mov_check.has_moved
    assert mov_check.mean_discrepancy_px < 1.0

    img_pts_perturbed = tuple((u + 5.0, v + 5.0) for u, v in img_pts_undisturbed)
    mov_check_moved = detect_camera_movement(
        nominal_pose=nominal_pose,
        object_points=obj_pts,
        image_points=img_pts_perturbed,
        intrinsics=intrinsics,
        threshold_px=1.5,
    )
    assert mov_check_moved.has_moved
    assert mov_check_moved.mean_discrepancy_px > 5.0


def test_evaluate_extrinsic_quality() -> None:
    q, deg = evaluate_extrinsic_quality(
        mean_residual_px=0.5,
        camera_count=4,
        min_points_per_view=30,
    )
    assert q is ExtrinsicQuality.QUALIFIED
    assert len(deg) == 0

    q_deg, _ = evaluate_extrinsic_quality(
        mean_residual_px=1.5,
        camera_count=2,
        min_points_per_view=10,
    )
    assert q_deg is ExtrinsicQuality.DEGRADED

    q_unq, deg_unq = evaluate_extrinsic_quality(
        mean_residual_px=5.0,
        camera_count=1,
        min_points_per_view=4,
    )
    assert q_unq is ExtrinsicQuality.UNQUALIFIED
    assert ExtrinsicDegeneracyKind.INSUFFICIENT_CAMERAS in deg_unq


def test_bundle_adjust_layout() -> None:
    world_frame = CoordinateFrame.affinedrift_world_v1()
    intrinsics = _sample_pinhole()
    t0 = RigidTransform(
        target_frame_id="cam_0",
        source_frame_id="world",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_m=(0.0, 0.0, 2.0),
    )
    t1 = RigidTransform(
        target_frame_id="cam_1",
        source_frame_id="world",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation_m=(1.0, 0.0, 2.0),
    )
    layout = CameraLayout(
        layout_id="layout_ba",
        world_frame=world_frame,
        camera_poses={
            "cam_0": CameraPose("cam_0", t0),
            "cam_1": CameraPose("cam_1", t1),
        },
    )
    obj_pts = ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.2, 0.2, 0.0), (0.0, 0.2, 0.0))
    obs_c0 = tuple(
        (pt, intrinsics.project_point((pt[0], pt[1], pt[2] + 2.0))) for pt in obj_pts
    )
    obs_c1 = tuple(
        (pt, intrinsics.project_point((pt[0] + 1.0, pt[1], pt[2] + 2.0)))
        for pt in obj_pts
    )

    res = bundle_adjust_layout(
        initial_layout=layout,
        observations_by_camera={"cam_0": obs_c0, "cam_1": obs_c1},
        intrinsics_by_camera={"cam_0": intrinsics, "cam_1": intrinsics},
    )
    assert res.mean_reprojection_residual_px < 1.0
    assert res.quality is not ExtrinsicQuality.UNQUALIFIED
