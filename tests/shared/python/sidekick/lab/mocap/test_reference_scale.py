"""Ruler scale needs an established camera layout (#5168)."""

from dataclasses import replace

import cv2
import numpy as np
import pytest
from sidekick.lab.mocap.calibration import (
    DistortionCoefficients,
    DistortionModel,
    PinholeIntrinsics,
)
from sidekick.lab.mocap.extrinsics import CameraLayout, CameraPose
from sidekick.lab.mocap.geometry import CoordinateFrame, RigidTransform
from sidekick.lab.mocap.reference_placements import (
    PlacementObservation,
    ReferenceCamera,
    common_reference,
)
from sidekick.lab.mocap.reference_scale import (
    ReferenceScaleOptions,
    ReferenceScaleProblem,
    estimate_reference_scale,
)


def problem() -> ReferenceScaleProblem:
    """OpenCV generates measurements independently of the scale implementation."""
    world = CoordinateFrame.affinedrift_world_v1()
    lens = PinholeIntrinsics(
        900,
        910,
        640,
        360,
        (1280, 720),
        DistortionCoefficients(
            DistortionModel.BROWN_CONRADY, (0.04, -0.02, 0.001, 0, 0)
        ),
    )
    matrix = np.array([[900, 0, 640], [0, 910, 360], [0, 0, 1.0]])
    reference = common_reference("yardstick")
    cameras = {key: ReferenceCamera(key, lens, f"{key}-zoom1") for key in ("a", "b")}
    centers = {"a": -0.5, "b": 0.5}
    poses = {
        key: CameraPose(
            key,
            RigidTransform(
                f"{key}-optical", world.frame_id, (1, 0, 0, 0), (-2 * center, 0, 0)
            ),
        )
        for key, center in centers.items()
    }
    observations = []
    for index in range(3):
        points = np.array(
            [
                [-0.3, 0.1 * index, 3.0 + index],
                [-0.3 + 0.9144, 0.1 * index, 3.0 + index],
            ]
        )
        for key, center in centers.items():
            pixels, _ = cv2.projectPoints(
                points,
                np.zeros(3),
                np.array([-center, 0, 0]),
                matrix,
                np.array(lens.distortion.coefficients),
            )
            observations.append(
                PlacementObservation(
                    f"p{index}",
                    reference.reference_id,
                    key,
                    f"{key}-zoom1",
                    reference.point_ids,
                    tuple(map(tuple, pixels.reshape(-1, 2))),
                    index,
                    index * 1_000_000,
                    held_out=index == 2,
                )
            )
    return ReferenceScaleProblem(
        CameraLayout("twice-actual-scale", world, poses),
        cameras,
        {reference.reference_id: reference},
        tuple(observations),
    )


def test_recovers_metric_scale_and_preserves_source_geometry() -> None:
    source = problem()
    before = dict(source.layout.camera_poses)
    result = estimate_reference_scale(source)
    assert result.scale_factor == pytest.approx(0.5, abs=1e-9)
    assert result.fit_placement_ids == ("p0", "p1")
    assert result.held_out_placement_ids == ("p2",)
    assert max(abs(row.relative_error) for row in result.residuals) < 1e-8
    assert result.scaled_layout.layout_id != source.layout.layout_id
    assert source.layout.camera_poses == before
    for key, pose in before.items():
        scaled = result.scaled_layout.get_pose(key)
        assert (
            scaled.t_camera_from_world.rotation_wxyz
            == pose.t_camera_from_world.rotation_wxyz
        )
        np.testing.assert_allclose(
            scaled.camera_center_world,
            np.array(pose.camera_center_world) / 2,
            atol=1e-9,
        )


def test_holdout_changes_validation_but_not_fitted_scale() -> None:
    source = problem()
    changed = []
    for observation in source.observations:
        if observation.held_out:
            pixels = np.array(observation.pixels_px)
            pixels[1, 0] += 0.2
            observation = replace(observation, pixels_px=tuple(map(tuple, pixels)))
        changed.append(observation)
    baseline = estimate_reference_scale(source)
    result = estimate_reference_scale(replace(source, observations=tuple(changed)))
    assert result.scale_factor == baseline.scale_factor
    assert next(row for row in result.residuals if row.held_out).relative_error != 0


@pytest.mark.parametrize(
    "invalid",
    ["profile", "duplicate", "one-camera", "all-holdout", "baseline", "units"],
)
def test_rejects_unusable_geometry_or_evidence(invalid: str) -> None:
    source = problem()
    observations = source.observations
    if invalid == "profile":
        observations = (
            replace(observations[0], profile_id="changed-zoom"),
            *observations[1:],
        )
    elif invalid == "duplicate":
        observations = (*observations, observations[0])
    elif invalid == "one-camera":
        observations = tuple(row for row in observations if row.camera_key == "a")
    elif invalid == "all-holdout":
        observations = tuple(replace(row, held_out=True) for row in observations)
    elif invalid == "baseline":
        poses = {
            key: replace(
                pose,
                t_camera_from_world=replace(
                    pose.t_camera_from_world, translation_m=(0, 0, 0)
                ),
            )
            for key, pose in source.layout.camera_poses.items()
        }
        source = replace(source, layout=replace(source.layout, camera_poses=poses))
    else:
        source = replace(
            source,
            layout=replace(
                source.layout,
                world_frame=replace(source.layout.world_frame, length_unit="cm"),
            ),
        )
    with pytest.raises(ValueError):
        estimate_reference_scale(replace(source, observations=observations))


def test_nonzero_anchor_preserves_rays_and_rotations() -> None:
    source = problem()
    anchor = np.array([0.2, -0.1, 0.3])
    result = estimate_reference_scale(
        source, ReferenceScaleOptions(anchor_m=tuple(anchor))
    )
    point = np.array([0.4, 0.3, 6.0])
    scaled_point = anchor + result.scale_factor * (point - anchor)
    for key, pose in source.layout.camera_poses.items():
        old_ray = point - np.array(pose.camera_center_world)
        new_ray = scaled_point - np.array(
            result.scaled_layout.get_pose(key).camera_center_world
        )
        np.testing.assert_allclose(
            new_ray / np.linalg.norm(new_ray),
            old_ray / np.linalg.norm(old_ray),
            atol=1e-10,
        )


def test_small_pixel_noise_has_bounded_scale_error_and_reported_repeatability() -> None:
    source = problem()
    rng = np.random.default_rng(5168)
    observations = tuple(
        replace(
            row,
            pixels_px=tuple(
                map(tuple, np.asarray(row.pixels_px) + rng.normal(0, 0.05, (2, 2)))
            ),
        )
        for row in source.observations
    )
    result = estimate_reference_scale(replace(source, observations=observations))
    assert result.scale_factor == pytest.approx(0.5, abs=0.001)
    assert result.scale_repeatability is not None and result.scale_repeatability > 0
    assert max(abs(row.relative_error) for row in result.residuals) < 0.01


def test_single_placement_reports_missing_repeatability_and_holdout() -> None:
    source = problem()
    result = estimate_reference_scale(
        replace(source, observations=source.observations[:2])
    )
    assert result.scale_factor == pytest.approx(0.5, abs=1e-9)
    assert result.scale_repeatability is None
    assert not result.held_out_placement_ids
    assert any("One fitting placement" in line for line in result.limitations)
    assert any("No independent" in line for line in result.limitations)
    with pytest.raises(TypeError):
        result.camera_profiles["a"] = "changed"
    with pytest.raises(TypeError):
        result.scaled_layout.camera_poses["a"] = source.layout.get_pose("a")


@pytest.mark.parametrize("placement", ["p1", "p2"])
def test_inconsistent_fit_or_holdout_length_is_rejected(placement: str) -> None:
    from sidekick.lab.mocap.reference_placements import ReferenceTarget

    source = problem()
    short = ReferenceTarget.line("incorrect-measured-length", 0.4)
    rows = tuple(
        replace(row, reference_id=short.reference_id)
        if row.placement_id == placement
        else row
        for row in source.observations
    )
    with pytest.raises(ValueError, match="length evidence"):
        estimate_reference_scale(
            replace(
                source,
                targets={**source.targets, short.reference_id: short},
                observations=rows,
            )
        )


def test_repeated_identical_placement_is_not_independent_evidence() -> None:
    source = problem()
    first = {
        row.camera_key: row.pixels_px
        for row in source.observations
        if row.placement_id == "p0"
    }
    rows = tuple(
        replace(row, pixels_px=first[row.camera_key])
        if row.placement_id == "p1"
        else row
        for row in source.observations
    )
    with pytest.raises(ValueError, match="independent placements"):
        estimate_reference_scale(replace(source, observations=rows))


def test_endpoint_order_follows_identity() -> None:
    source = problem()
    rows = tuple(
        replace(
            row,
            point_ids=tuple(reversed(row.point_ids)),
            pixels_px=tuple(reversed(row.pixels_px)),
        )
        for row in source.observations
    )
    result = estimate_reference_scale(replace(source, observations=rows))
    assert result.scale_factor == pytest.approx(0.5, abs=1e-9)
