"""Swept wedge-to-planar-ground contact contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.python.golf_club import (
    ContactSequence,
    GroundPlane,
    WedgeContactFeature,
    WedgePreset,
    analyze_wedge_ground_clearance,
    wedge_contact_candidates,
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
