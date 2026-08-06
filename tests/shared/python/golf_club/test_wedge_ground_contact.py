"""Swept wedge-to-planar-ground contact contracts."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club import (
    ContactSequence,
    GroundPlane,
    Handedness,
    WedgeContactFeature,
    WedgePreset,
    analyze_wedge_ground_clearance,
    wedge_contact_candidates,
    wedge_face_contact_point_m,
    wedge_preset,
)


def _pose(*, height_m: float, roll_deg: float = 0.0) -> np.ndarray:
    angle = math.radians(roll_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
    )
    pose[1, 3] = height_m
    return pose


def _sweep(
    poses: np.ndarray,
    *,
    ball_contact_time_s: float | None = None,
    times_s: tuple[float, ...] = (0.0, 1.0),
):
    twists = np.zeros((len(times_s), 6))
    twists[:, 4] = (poses[-1, 1, 3] - poses[0, 1, 3]) / (times_s[-1] - times_s[0])
    return analyze_wedge_ground_clearance(
        wedge_preset(WedgePreset.MID_BOUNCE),
        times_s,
        poses,
        twists,
        GroundPlane(),
        ball_contact_time_s=ball_contact_time_s,
    )


def test_contact_candidates_cover_leading_primary_and_trailing_regions() -> None:
    low = wedge_preset(WedgePreset.LOW_BOUNCE)
    high = wedge_preset(WedgePreset.HIGH_BOUNCE)

    low_candidates = wedge_contact_candidates(low)
    high_candidates = wedge_contact_candidates(high)

    assert {candidate.feature for candidate in low_candidates} == set(
        WedgeContactFeature
    )
    assert len(low_candidates) == 9
    low_trailing = next(
        candidate
        for candidate in low_candidates
        if candidate.feature is WedgeContactFeature.TRAILING_SOLE_CENTER
    )
    high_trailing = next(
        candidate
        for candidate in high_candidates
        if candidate.feature is WedgeContactFeature.TRAILING_SOLE_CENTER
    )
    assert high_trailing.local_point_m[1] > low_trailing.local_point_m[1]


def test_handedness_mirrors_heel_and_toe_candidate_datums() -> None:
    right = wedge_preset(WedgePreset.MID_BOUNCE)
    left = replace(right, handedness=Handedness.LEFT)

    right_by_feature = {
        candidate.feature: candidate.local_point_m
        for candidate in wedge_contact_candidates(right)
    }
    left_by_feature = {
        candidate.feature: candidate.local_point_m
        for candidate in wedge_contact_candidates(left)
    }

    for right_feature, left_feature in (
        (WedgeContactFeature.LEADING_EDGE_HEEL, WedgeContactFeature.LEADING_EDGE_HEEL),
        (WedgeContactFeature.LEADING_EDGE_TOE, WedgeContactFeature.LEADING_EDGE_TOE),
        (WedgeContactFeature.PRIMARY_SOLE_HEEL, WedgeContactFeature.PRIMARY_SOLE_HEEL),
        (WedgeContactFeature.TRAILING_SOLE_TOE, WedgeContactFeature.TRAILING_SOLE_TOE),
    ):
        right_point = right_by_feature[right_feature]
        left_point = left_by_feature[left_feature]
        assert left_point[:2] == pytest.approx(right_point[:2])
        assert left_point[2] == pytest.approx(-right_point[2])


def test_face_contact_point_uses_lofted_high_and_handed_toe_offsets() -> None:
    right = wedge_preset(WedgePreset.MID_BOUNCE)
    left = replace(right, handedness=Handedness.LEFT)
    center = np.asarray(wedge_face_contact_point_m(right, 0.0, 0.0))
    high = np.asarray(wedge_face_contact_point_m(right, 0.0, 0.005))
    right_toe = wedge_face_contact_point_m(right, 0.010, 0.0)
    left_toe = wedge_face_contact_point_m(left, 0.010, 0.0)
    face_tangent = np.array(
        [
            -math.sin(math.radians(right.loft_deg)),
            math.cos(math.radians(right.loft_deg)),
            0.0,
        ]
    )

    np.testing.assert_allclose(high - center, 0.005 * face_tangent, atol=1e-12)
    assert right_toe[2] == pytest.approx(0.010)
    assert left_toe[2] == pytest.approx(-0.010)


def test_face_contact_point_rejects_offsets_outside_the_face() -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)

    with pytest.raises(ValueError, match="toe_offset_m"):
        wedge_face_contact_point_m(parameters, parameters.face_length_m, 0.0)
    with pytest.raises(ValueError, match="high_offset_m"):
        wedge_face_contact_point_m(parameters, 0.0, parameters.face_height_m)


def test_between_frame_crossing_is_refined_to_the_analytic_time() -> None:
    poses = np.stack([_pose(height_m=0.01), _pose(height_m=-0.01)])

    result = _sweep(poses, ball_contact_time_s=0.25)

    assert result.first_ground_contact is not None
    assert result.first_ground_contact.time_s == pytest.approx(0.5, abs=1e-9)
    assert (
        result.first_ground_contact.feature is WedgeContactFeature.LEADING_EDGE_CENTER
    )
    assert result.first_ground_contact.normal_velocity_mps == pytest.approx(-0.02)
    assert result.first_ground_contact.tangential_velocity_mps == pytest.approx(
        (0.0, 0.0, 0.0)
    )
    assert result.sequence is ContactSequence.BALL_FIRST
    assert result.ground_after_ball_time_margin_s == pytest.approx(0.25)


def test_nonlevel_lie_selects_the_low_toe_contact_region() -> None:
    poses = np.stack(
        [_pose(height_m=0.01, roll_deg=20.0), _pose(height_m=0.01, roll_deg=20.0)]
    )

    result = _sweep(poses)

    assert result.first_ground_contact is not None
    assert result.first_ground_contact.time_s == pytest.approx(0.0)
    assert result.first_ground_contact.feature is WedgeContactFeature.LEADING_EDGE_TOE
    assert result.sequence is ContactSequence.GROUND_ONLY_MISS


@pytest.mark.parametrize(
    ("ball_time", "end_height", "expected"),
    [
        (0.75, -0.01, ContactSequence.GROUND_FIRST),
        (0.50, -0.01, ContactSequence.SIMULTANEOUS),
        (0.50, 0.01, ContactSequence.BALL_ONLY),
        (None, 0.01, ContactSequence.NO_CONTACT_MISS),
    ],
)
def test_contact_sequence_is_explicit_for_hits_and_misses(
    ball_time: float | None,
    end_height: float,
    expected: ContactSequence,
) -> None:
    poses = np.stack([_pose(height_m=0.01), _pose(height_m=end_height)])

    result = _sweep(poses, ball_contact_time_s=ball_time)

    assert result.sequence is expected


def test_ball_contact_metrics_report_clearance_low_point_and_bounce() -> None:
    parameters = wedge_preset(WedgePreset.HIGH_BOUNCE)
    poses = np.stack([_pose(height_m=0.02), _pose(height_m=0.01)])
    twists = np.zeros((2, 6))
    twists[:, 3] = 10.0
    twists[:, 4] = -0.01

    result = analyze_wedge_ground_clearance(
        parameters,
        (0.0, 1.0),
        poses,
        twists,
        GroundPlane(),
        ball_contact_time_s=0.5,
    )

    assert result.leading_edge_clearance_at_ball_m == pytest.approx(0.015)
    assert result.minimum_pre_ball_clearance_m == pytest.approx(0.015)
    assert result.low_point_time_s == pytest.approx(1.0)
    assert result.low_point_world_m[1] == pytest.approx(0.01)
    assert result.delivered_bounce_deg_at_ball == pytest.approx(parameters.bounce_deg)
    assert result.sole_entry_margin_m == pytest.approx(
        0.015
        + parameters.leading_edge_radius_m
        + 0.5 * parameters.sole_width_m * math.sin(math.radians(parameters.bounce_deg))
    )
    assert result.path_projected_effective_bounce_deg_at_ball == pytest.approx(
        parameters.bounce_deg
    )
    expected_aoa = math.degrees(math.atan2(-0.01, 10.0))
    assert result.reference_aoa_deg_at_ball == pytest.approx(expected_aoa)
    assert result.bounce_utilization_margin_deg == pytest.approx(
        parameters.bounce_deg + expected_aoa
    )


def test_path_projected_metrics_are_unavailable_without_horizontal_motion() -> None:
    poses = np.stack([_pose(height_m=0.02), _pose(height_m=0.01)])

    result = _sweep(poses, ball_contact_time_s=0.5)

    assert result.path_projected_effective_bounce_deg_at_ball is None
    assert result.reference_aoa_deg_at_ball is None
    assert result.bounce_utilization_margin_deg is None


def test_invalid_pose_and_time_contracts_fail_actionably() -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    poses = np.stack([_pose(height_m=0.01), _pose(height_m=0.0)])
    twists = np.zeros((2, 6))

    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_wedge_ground_clearance(
            parameters, (0.0, 0.0), poses, twists, GroundPlane()
        )
    with pytest.raises(ValueError, match="rotation"):
        invalid = poses.copy()
        invalid[1, 0, 0] = 2.0
        analyze_wedge_ground_clearance(
            parameters, (0.0, 1.0), invalid, twists, GroundPlane()
        )


def test_clearance_is_invariant_when_sweep_and_ground_translate_together() -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    poses = np.stack([_pose(height_m=0.01), _pose(height_m=-0.01)])
    twists = np.zeros((2, 6))
    twists[:, 4] = -0.02
    baseline = analyze_wedge_ground_clearance(
        parameters,
        (0.0, 1.0),
        poses,
        twists,
        GroundPlane(),
        ball_contact_time_s=0.25,
    )
    offset = np.array((1.25, -0.4, 2.5))
    translated_poses = poses.copy()
    translated_poses[:, :3, 3] += offset

    translated = analyze_wedge_ground_clearance(
        parameters,
        (0.0, 1.0),
        translated_poses,
        twists,
        GroundPlane(point_m=tuple(offset)),
        ball_contact_time_s=0.25,
    )

    assert translated.sequence is baseline.sequence
    assert translated.first_ground_contact is not None
    assert baseline.first_ground_contact is not None
    assert translated.first_ground_contact.time_s == pytest.approx(
        baseline.first_ground_contact.time_s
    )
    assert [sample.minimum_clearance_m for sample in translated.envelope] == (
        pytest.approx([sample.minimum_clearance_m for sample in baseline.envelope])
    )
    assert np.asarray(translated.low_point_world_m) == pytest.approx(
        np.asarray(baseline.low_point_world_m) + offset
    )


def test_analysis_is_equivariant_under_common_rigid_frame_rotation() -> None:
    parameters = wedge_preset(WedgePreset.MID_BOUNCE)
    poses = np.stack([_pose(height_m=0.01), _pose(height_m=-0.01)])
    twists = np.zeros((2, 6))
    twists[:, 4] = -0.02
    baseline = analyze_wedge_ground_clearance(
        parameters,
        (0.0, 1.0),
        poses,
        twists,
        GroundPlane(),
        ball_contact_time_s=0.25,
    )
    angle = math.radians(37.0)
    cosine, sine = math.cos(angle), math.sin(angle)
    frame_rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated_poses = poses.copy()
    rotated_poses[:, :3, :3] = frame_rotation @ poses[:, :3, :3]
    rotated_poses[:, :3, 3] = (frame_rotation @ poses[:, :3, 3].T).T
    rotated_twists = twists.copy()
    rotated_twists[:, :3] = (frame_rotation @ twists[:, :3].T).T
    rotated_twists[:, 3:] = (frame_rotation @ twists[:, 3:].T).T

    rotated = analyze_wedge_ground_clearance(
        parameters,
        (0.0, 1.0),
        rotated_poses,
        rotated_twists,
        GroundPlane(normal_unit=tuple(frame_rotation @ np.array((0.0, 1.0, 0.0)))),
        ball_contact_time_s=0.25,
    )

    assert rotated.first_ground_contact is not None
    assert baseline.first_ground_contact is not None
    assert rotated.first_ground_contact.feature is baseline.first_ground_contact.feature
    assert rotated.first_ground_contact.time_s == pytest.approx(
        baseline.first_ground_contact.time_s
    )
    assert np.asarray(rotated.first_ground_contact.world_point_m) == pytest.approx(
        frame_rotation @ np.asarray(baseline.first_ground_contact.world_point_m)
    )


def test_time_origin_shift_preserves_geometry_and_shifts_event_time() -> None:
    poses = np.stack([_pose(height_m=0.01), _pose(height_m=-0.01)])

    baseline = _sweep(poses, ball_contact_time_s=0.25)
    shifted = _sweep(
        poses,
        ball_contact_time_s=2.25,
        times_s=(2.0, 3.0),
    )

    assert shifted.first_ground_contact is not None
    assert baseline.first_ground_contact is not None
    assert shifted.first_ground_contact.time_s == pytest.approx(
        baseline.first_ground_contact.time_s + 2.0
    )
    assert shifted.sequence is baseline.sequence
    assert shifted.ground_after_ball_time_margin_s == pytest.approx(
        baseline.ground_after_ball_time_margin_s
    )
    assert [sample.minimum_clearance_m for sample in shifted.envelope] == (
        pytest.approx([sample.minimum_clearance_m for sample in baseline.envelope])
    )


def test_linear_contact_time_is_stable_under_retained_timestep_refinement() -> None:
    coarse = np.stack([_pose(height_m=0.01), _pose(height_m=-0.01)])
    refined = np.stack(
        [
            _pose(height_m=0.01),
            _pose(height_m=0.0),
            _pose(height_m=-0.01),
        ]
    )

    coarse_result = _sweep(coarse)
    refined_result = _sweep(refined, times_s=(0.0, 0.5, 1.0))

    assert coarse_result.first_ground_contact is not None
    assert refined_result.first_ground_contact is not None
    assert coarse_result.first_ground_contact.time_s == pytest.approx(0.5, abs=1e-9)
    assert refined_result.first_ground_contact.time_s == pytest.approx(
        coarse_result.first_ground_contact.time_s,
        abs=1e-9,
    )
