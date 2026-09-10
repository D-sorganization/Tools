"""Independent numerical recovery and failure evidence for calibration (#5132)."""

from __future__ import annotations

import sys
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sidekick.lab.mocap.calibration import (
    DistortionCoefficients,
    DistortionModel,
    FisheyeIntrinsics,
    PinholeIntrinsics,
)
from sidekick.lab.mocap.extrinsics import (
    CameraLayout,
    CameraPose,
    bundle_adjust_layout,
    estimate_pnp_pose,
)
from sidekick.lab.mocap.geometry import CoordinateFrame, RigidTransform


def camera(model: DistortionModel = DistortionModel.NONE) -> PinholeIntrinsics:
    coefficients = {
        DistortionModel.NONE: (),
        DistortionModel.BROWN_CONRADY: (0.12, -0.06, 0.003, -0.002, 0.01),
        DistortionModel.RATIONAL: (0.12, -0.06, 0.003, -0.002, 0.01, 0.04, 0.02, 0.001),
    }
    return PinholeIntrinsics(
        900,
        930,
        960,
        540,
        (1920, 1080),
        DistortionCoefficients(model, coefficients[model]),
    )


def points() -> np.ndarray:
    return np.array(
        [
            (x, y, z)
            for x in (-0.3, 0, 0.3)
            for y in (-0.2, 0.1, 0.4)
            for z in (0, 0.15)
        ],
        dtype=float,
    )


def pixels(
    intrinsics: PinholeIntrinsics,
    world: np.ndarray,
    rvec: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    k = np.array(
        [
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1],
        ]
    )
    projected, _ = cv2.projectPoints(
        world,
        rvec,
        translation,
        k,
        np.array(intrinsics.distortion.coefficients)
        if intrinsics.distortion.coefficients
        else None,
    )
    return projected.reshape(-1, 2)


@pytest.mark.parametrize(
    "model", [DistortionModel.BROWN_CONRADY, DistortionModel.RATIONAL]
)
def test_distorted_projection_and_inverse_ray_match_independent_opencv(
    model: DistortionModel,
) -> None:
    intrinsics = camera(model)
    xyz = np.array([[0.8, 0.6, 1.2]])
    expected = pixels(intrinsics, xyz, np.zeros(3), np.zeros(3))[0]
    assert intrinsics.project_point(tuple(xyz[0])) == pytest.approx(expected, abs=1e-8)
    ray = np.array(intrinsics.unproject_point(tuple(expected)))
    assert ray == pytest.approx(xyz[0] / np.linalg.norm(xyz[0]), abs=1e-8)


def test_nonzero_fisheye_distortion_round_trip() -> None:
    intrinsics = FisheyeIntrinsics(
        900,
        930,
        960,
        540,
        (1920, 1080),
        DistortionCoefficients(
            DistortionModel.KANNALA_BRANDT, (0.12, -0.03, 0.01, 0.002)
        ),
    )
    xyz = np.array([0.8, 0.6, 1.2])
    ray = intrinsics.unproject_point(intrinsics.project_point(tuple(xyz)))
    assert ray == pytest.approx(xyz / np.linalg.norm(xyz), abs=1e-8)


def test_solver_failure_cannot_become_a_fixed_camera_pose() -> None:
    intrinsics = camera()
    world = points()
    observed = pixels(intrinsics, world, np.zeros(3), np.array([0, 0, 2.0]))
    with patch.object(
        cv2, "solvePnP", side_effect=cv2.error("injected solver failure")
    ):
        with pytest.raises(RuntimeError, match="solve|PnP"):
            estimate_pnp_pose("cam", world, observed, intrinsics, "cam", "world")


def test_collinear_reference_points_are_rejected() -> None:
    world = [(i * 0.1, 0, 0) for i in range(6)]
    observed = [(960 + i * 45.0, 540) for i in range(6)]
    with pytest.raises(ValueError, match="collinear|degenerate"):
        estimate_pnp_pose("cam", world, observed, camera(), "cam", "world")


@pytest.mark.parametrize("fault", ["missing", "unsuccessful", "nonfinite"])
def test_missing_solver_and_invalid_inputs_cannot_produce_calibration(
    fault: str,
) -> None:
    intrinsics = camera()
    world = points()
    observed = pixels(intrinsics, world, np.zeros(3), np.array([0, 0, 2.0]))
    arguments = ("cam", world, observed, intrinsics, "cam", "world")
    if fault == "missing":
        with patch.dict(sys.modules, {"cv2": None}), pytest.raises(ImportError):
            estimate_pnp_pose(*arguments)
    elif fault == "unsuccessful":
        with (
            patch.object(cv2, "solvePnP", return_value=(False, None, None)),
            pytest.raises(RuntimeError, match="converge"),
        ):
            estimate_pnp_pose(*arguments)
    else:
        world[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            estimate_pnp_pose(*arguments)


def test_rational_skew_projection_and_inverse_are_consistent() -> None:
    intrinsics = replace(camera(DistortionModel.RATIONAL), skew=11)
    xyz = np.array([[0.8, 0.6, 1.2]])
    expected = pixels(intrinsics, xyz, np.zeros(3), np.zeros(3))[0]
    expected[0] += intrinsics.skew * (expected[1] - intrinsics.cy) / intrinsics.fy
    assert intrinsics.project_point(tuple(xyz[0])) == pytest.approx(expected, abs=1e-8)
    assert intrinsics.unproject_point(tuple(expected)) == pytest.approx(
        xyz[0] / np.linalg.norm(xyz[0]), abs=1e-8
    )


def test_explicit_gauge_stays_fixed_and_missing_views_are_errors() -> None:
    intrinsics = camera()
    pose = CameraPose("cam", RigidTransform("cam", "world", (1, 0, 0, 0), (0, 0, 2)))
    frame = replace(CoordinateFrame.affinedrift_world_v1(), frame_id="world")
    layout = CameraLayout("test", frame, {"cam": pose})
    observed = pixels(intrinsics, points(), np.zeros(3), np.array([0.02, 0, 2.0]))
    observations = {
        "cam": tuple(
            (tuple(xyz), tuple(uv)) for xyz, uv in zip(points(), observed, strict=True)
        )
    }
    result = bundle_adjust_layout(layout, observations, {"cam": intrinsics}, "cam")
    assert result.layout.get_pose("cam") == pose
    assert result.mean_reprojection_residual_px > 1
    with pytest.raises(ValueError, match="every layout camera"):
        bundle_adjust_layout(layout, {}, {"cam": intrinsics})
    with pytest.raises(ValueError, match="no calibration observations"):
        bundle_adjust_layout(layout, {"cam": ()}, {"cam": intrinsics})
    behind = replace(
        pose,
        t_camera_from_world=replace(pose.t_camera_from_world, translation_m=(0, 0, -2)),
    )
    with pytest.raises(ValueError, match="in front"):
        bundle_adjust_layout(
            replace(layout, camera_poses={"cam": behind}),
            observations,
            {"cam": intrinsics},
        )


def test_fisheye_pose_recovers_nonzero_distortion() -> None:
    intrinsics = FisheyeIntrinsics(
        900,
        930,
        960,
        540,
        (1920, 1080),
        DistortionCoefficients(
            DistortionModel.KANNALA_BRANDT, (0.12, -0.03, 0.01, 0.002)
        ),
    )
    world = points()
    rotation = Rotation.from_rotvec([0.1, -0.15, 0.03])
    translation = np.array([0.1, 0.2, 2])
    observed = [
        intrinsics.project_point(tuple(xyz))
        for xyz in rotation.apply(world) + translation
    ]
    pose, residual = estimate_pnp_pose(
        "cam", world, observed, intrinsics, "cam", "world"
    )
    assert pose.t_camera_from_world.translation_m == pytest.approx(
        translation, abs=1e-6
    )
    assert residual < 1e-6


def test_bundle_adjustment_recovers_perturbed_cameras_and_unseen_points() -> None:
    intrinsics = camera(DistortionModel.RATIONAL)
    world = points()
    poses, observations, truths = {}, {}, {}
    for index in range(3):
        key = f"cam{index}"
        rvec = np.array([0.08 * index, -0.1, 0.03])
        translation = np.array([-0.2 + index * 0.2, 0.05 * index, 2.3])
        truths[key] = (rvec, translation)
        xyzw = Rotation.from_rotvec(rvec + [0.02, 0.03, -0.01]).as_quat()
        rotation = xyzw[[3, 0, 1, 2]]
        transform = RigidTransform(
            key, "world", tuple(rotation), tuple(translation + [0.04, -0.03, 0.1])
        )
        poses[key] = CameraPose(key, transform)
        observed = pixels(intrinsics, world, rvec, translation)
        observations[key] = tuple(
            (tuple(xyz), tuple(uv)) for xyz, uv in zip(world, observed, strict=True)
        )
    frame = replace(CoordinateFrame.affinedrift_world_v1(), frame_id="world")
    initial = CameraLayout("test", frame, poses)
    result = bundle_adjust_layout(
        initial, observations, dict.fromkeys(poses, intrinsics)
    )
    assert result.mean_reprojection_residual_px < 1e-5
    held_out = np.array([[0.15, -0.1, 0.07], [-0.17, 0.23, 0.1]])
    for key, (rvec, translation) in truths.items():
        fitted = result.layout.get_pose(key).t_camera_from_world
        assert fitted.translation_m == pytest.approx(translation, abs=1e-5)
        w, x, y, z = fitted.rotation_wxyz
        recovered = Rotation.from_quat([x, y, z, w]).as_rotvec()
        assert pixels(
            intrinsics, held_out, recovered, np.array(fitted.translation_m)
        ) == pytest.approx(pixels(intrinsics, held_out, rvec, translation), abs=1e-5)
        assert initial.get_pose(key).t_camera_from_world != fitted
