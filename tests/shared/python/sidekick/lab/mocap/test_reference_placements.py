"""Independent multi-placement camera recovery evidence (#5137)."""

from dataclasses import replace

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sidekick.lab.mocap.calibration import (
    DistortionCoefficients,
    DistortionModel,
    PinholeIntrinsics,
)
from sidekick.lab.mocap.geometry import CoordinateFrame, RigidTransform
from sidekick.lab.mocap.reference_placements import (
    PlacementObservation,
    ReferenceCamera,
    ReferenceSolveCancelled,
    ReferenceTarget,
    common_reference,
    estimate_reference_layout,
)


def fixture():
    target = common_reference("us-letter")
    lens = PinholeIntrinsics(
        1200,
        1180,
        960,
        540,
        (1920, 1080),
        DistortionCoefficients(
            DistortionModel.BROWN_CONRADY, (0.04, -0.02, 0.001, -0.002, 0.0)
        ),
    )
    cameras = {
        key: ReferenceCamera(key, lens, f"{key}-zoom1") for key in ("a", "b", "c")
    }
    # Independent transforms/projection use SciPy/OpenCV directly, not the solver.
    camera_poses = {
        "a": (
            Rotation.from_euler("xyz", [65, 0, 0], degrees=True),
            np.array([0, 0, 2.0]),
        ),
        "b": (
            Rotation.from_euler("xyz", [60, -15, 5], degrees=True),
            np.array([0.4, 0.1, 2.3]),
        ),
        "c": (
            Rotation.from_euler("xyz", [75, 20, -4], degrees=True),
            np.array([-0.4, -0.1, 2.1]),
        ),
    }
    placements = {
        "p0": (Rotation.from_euler("y", 13, degrees=True), np.array([0.1, 0.03, -0.1])),
        "p1": (
            Rotation.from_euler("xyz", [8, 35, -7], degrees=True),
            np.array([0.4, 0.05, 0.3]),
        ),
        "p2": (
            Rotation.from_euler("xyz", [-12, -30, 10], degrees=True),
            np.array([-0.35, 0.2, 0.25]),
        ),
        "p3": (
            Rotation.from_euler("xyz", [20, 60, 15], degrees=True),
            np.array([0.15, 0.35, -0.3]),
        ),
    }
    observations = []
    k = np.array([[lens.fx, 0, lens.cx], [0, lens.fy, lens.cy], [0, 0, 1.0]])
    for index, (placement, (r, t)) in enumerate(placements.items()):
        world = r.apply(target.object_points_m) + t
        for camera, (cr, ct) in camera_poses.items():
            pixels, _ = cv2.projectPoints(
                world, cr.as_rotvec(), ct, k, np.array(lens.distortion.coefficients)
            )
            observations.append(
                PlacementObservation(
                    placement,
                    target.reference_id,
                    camera,
                    f"{camera}-zoom1",
                    target.point_ids,
                    tuple(map(tuple, pixels.reshape(-1, 2))),
                    index,
                    index * 1000000,
                    held_out=(placement, camera)
                    in {("p1", "a"), ("p2", "b"), ("p3", "c")},
                )
            )
    r, t = placements["p0"]
    x, y, z, w = r.as_quat()
    anchor = RigidTransform(
        "affinedrift-world-v1", "placement:p0", (w, x, y, z), tuple(t)
    )
    return target, cameras, observations, anchor, camera_poses, placements


def solve(target, cameras, observations, anchor):
    return estimate_reference_layout(
        layout_id="practice",
        world_frame=CoordinateFrame.affinedrift_world_v1(),
        targets={target.reference_id: target},
        cameras=cameras,
        observations=observations,
        anchor_placement_id="p0",
        anchor=anchor,
    )


def test_common_references_preserve_dimensions_and_identity():
    paper = common_reference("us-letter")
    assert np.ptp(np.array(paper.object_points_m), axis=0) == pytest.approx(
        [0.2794, 0, 0.2159]
    )
    assert common_reference("a4").object_points_m[2] == pytest.approx((0.297, 0, 0.210))
    yard = common_reference("yardstick")
    assert np.linalg.norm(np.subtract(*yard.object_points_m)) == pytest.approx(0.9144)
    assert common_reference("meter-stick").object_points_m[1] == (1.0, 0.0, 0.0)
    assert ReferenceTarget.rectangle("custom", 0.3, 0.5).object_points_m[2] == (
        0.5,
        0.0,
        0.3,
    )
    with pytest.raises(ValueError):
        common_reference("unknown")
    with pytest.raises(ValueError):
        ReferenceTarget.rectangle("bad", float("nan"), 0.2)


def test_recovers_cameras_and_distinct_placements_with_held_out_views():
    target, cameras, observations, anchor, expected_cameras, expected_targets = (
        fixture()
    )
    result = solve(target, cameras, observations, anchor)
    assert max(item.max_error_px for item in result.residuals) < 1e-5
    assert len([item for item in result.residuals if item.held_out]) == 3
    assert result.anchor_placement_id == "p0"
    for key, (rotation, translation) in expected_cameras.items():
        actual = result.layout.get_pose(key).t_camera_from_world
        assert actual.translation_m == pytest.approx(translation, abs=1e-6)
        q = actual.rotation_wxyz
        recovered = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        assert (recovered.inv() * rotation).magnitude() < 1e-6
    for key, (_, translation) in expected_targets.items():
        assert result.placement_transforms[key].translation_m == pytest.approx(
            translation, abs=1e-6
        )
    with pytest.raises(TypeError):
        result.placement_transforms["p0"] = anchor


def test_holdout_errors_do_not_change_the_fit():
    target, cameras, observations, anchor, _, _ = fixture()
    damaged = [
        replace(obs, pixels_px=tuple((u + 15, v - 4) for u, v in obs.pixels_px))
        if obs.held_out
        else obs
        for obs in observations
    ]
    result = solve(target, cameras, damaged, anchor)
    assert (
        max(item.max_error_px for item in result.residuals if not item.held_out) < 1e-5
    )
    assert min(item.mean_error_px for item in result.residuals if item.held_out) > 15
    assert "Held-out observations exceed" in " ".join(result.limitations)


def test_rejects_disconnected_duplicate_and_stale_profile_observations():
    target, cameras, observations, anchor, _, _ = fixture()
    disconnected = [
        o for o in observations if (o.camera_key == "a") == (o.placement_id == "p0")
    ]
    with pytest.raises(ValueError, match="connected"):
        solve(target, cameras, disconnected, anchor)
    with pytest.raises(ValueError, match="duplicate"):
        solve(target, cameras, observations + [observations[0]], anchor)
    with pytest.raises(ValueError, match="profile"):
        solve(
            target,
            cameras,
            [replace(observations[0], profile_id="changed-zoom"), *observations[1:]],
            anchor,
        )


def test_rejects_wrong_point_identity_and_out_of_frame_pixels():
    target, cameras, observations, anchor, _, _ = fixture()
    with pytest.raises(ValueError, match="point"):
        solve(
            target,
            cameras,
            [
                replace(observations[0], point_ids=("bad", *target.point_ids[1:])),
                *observations[1:],
            ],
            anchor,
        )
    with pytest.raises(ValueError, match="image"):
        solve(
            target,
            cameras,
            [
                replace(
                    observations[0],
                    pixels_px=((-1.0, 50.0), *observations[0].pixels_px[1:]),
                ),
                *observations[1:],
            ],
            anchor,
        )
    with pytest.raises(ValueError, match="finite"):
        replace(
            observations[0],
            pixels_px=((float("nan"), 50.0), *observations[0].pixels_px[1:]),
        )


def test_failed_optimizer_is_an_error(monkeypatch):
    from types import SimpleNamespace

    import scipy.optimize

    target, cameras, observations, anchor, _, _ = fixture()
    monkeypatch.setattr(
        scipy.optimize, "least_squares", lambda *a, **k: SimpleNamespace(success=False)
    )
    with pytest.raises(RuntimeError, match="converge"):
        solve(target, cameras, observations, anchor)


def test_line_reference_cannot_initialize_a_full_camera_pose():
    target, cameras, observations, anchor, _, _ = fixture()
    line = common_reference("yardstick")
    observations = [
        replace(
            o,
            reference_id=line.reference_id,
            point_ids=line.point_ids,
            pixels_px=o.pixels_px[:2],
        )
        for o in observations
    ]
    with pytest.raises(ValueError, match="four non-collinear"):
        solve(line, cameras, observations, anchor)


def test_noisy_observations_recover_metric_camera_centers():
    target, cameras, observations, anchor, expected, _ = fixture()
    rng = np.random.default_rng(5137)
    noisy = [
        replace(
            o,
            pixels_px=tuple(
                map(tuple, np.asarray(o.pixels_px) + rng.normal(0, 0.08, (4, 2)))
            ),
        )
        for o in observations
    ]
    result = solve(target, cameras, noisy, anchor)
    for key, (rotation, translation) in expected.items():
        center = -rotation.inv().apply(translation)
        assert result.layout.get_pose(key).camera_center_world == pytest.approx(
            center, abs=0.025
        )
    assert max(item.max_error_px for item in result.residuals) < 0.5
    assert result.observations == tuple(noisy)
    assert result.camera_profiles == {key: f"{key}-zoom1" for key in cameras}
    with pytest.raises(TypeError):
        result.layout.camera_poses["a"] = result.layout.get_pose("b")


def test_same_placement_images_are_not_independent_evidence():
    target, cameras, observations, anchor, _, _ = fixture()
    first = [o for o in observations if o.placement_id == "p0"]
    repeated = first + [replace(o, placement_id="repeat") for o in first]
    with pytest.raises(ValueError, match="independent placements"):
        solve(target, cameras, repeated, anchor)


def test_cancel_during_optimization_does_not_return_a_layout():
    target, cameras, observations, anchor, _, _ = fixture()
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 20

    with pytest.raises(ReferenceSolveCancelled):
        estimate_reference_layout(
            layout_id="cancelled",
            world_frame=CoordinateFrame.affinedrift_world_v1(),
            targets={target.reference_id: target},
            cameras=cameras,
            observations=observations,
            anchor_placement_id="p0",
            anchor=anchor,
            cancel_requested=cancelled,
        )


def test_target_point_order_and_input_mutation_are_preserved():
    target, _, observations, _, _, _ = fixture()
    observation = observations[0]
    pixels = list(map(list, observation.pixels_px))
    saved = replace(
        observation, pixels_px=pixels, point_ids=list(observation.point_ids)
    )
    pixels[0][0] = 0
    assert saved == observation
    reversed_observation = replace(
        observation,
        point_ids=observation.point_ids[::-1],
        pixels_px=observation.pixels_px[::-1],
    )
    canonical = reversed_observation.calibration_observation(target)
    assert canonical.target_object_points == target.object_points_m[::-1]
    assert canonical.detection_confidence == 0
    with pytest.raises(ValueError):
        replace(observation, point_ids=("origin",) * 4)
    with pytest.raises(ValueError):
        replace(observation, detection_confidence=2)


def test_camera_need_not_see_anchor_if_other_placements_connect_it():
    target, cameras, observations, anchor, expected, _ = fixture()
    edges = {
        ("a", "p0"),
        ("a", "p1"),
        ("b", "p1"),
        ("b", "p2"),
        ("c", "p2"),
        ("c", "p3"),
    }
    chain = [
        replace(o, held_out=False)
        for o in observations
        if (o.camera_key, o.placement_id) in edges
    ]
    result = solve(target, cameras, chain, anchor)
    for key, (_, translation) in expected.items():
        assert result.layout.get_pose(
            key
        ).t_camera_from_world.translation_m == pytest.approx(translation, abs=1e-5)
    assert sum("no held-out view" in limit for limit in result.limitations) == 3
