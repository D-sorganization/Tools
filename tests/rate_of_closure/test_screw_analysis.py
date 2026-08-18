"""Scientific contracts for club and articulated-joint screw analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rate_of_closure.club import club_names, get_club
from rate_of_closure.model import ImpactScenario
from rate_of_closure.simulation import SimulationConfig, run_simulation
from rate_of_closure.simulation.screw_analysis import (
    MotionKind,
    analyze_joint_motion,
    analyze_twist,
    build_screw_glyph,
    project_motion,
)


def test_finite_screw_reconstructs_reference_velocity() -> None:
    omega = np.array([0.0, 0.0, 4.0])
    pitch_m_rad = 0.125
    axis_point = np.array([1.0, -2.0, 0.5])
    reference = np.array([2.0, -2.0, 0.5])
    velocity = np.cross(omega, reference - axis_point) + pitch_m_rad * omega

    motion = analyze_twist(np.concatenate([omega, velocity]), reference)

    assert motion.kind is MotionKind.FINITE
    np.testing.assert_allclose(motion.axis_direction, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(motion.axis_point_m, axis_point)
    assert motion.pitch_m_rad == pytest.approx(pitch_m_rad)
    assert motion.radius_m == pytest.approx(1.0)
    np.testing.assert_allclose(
        motion.orbital_velocity_m_s + motion.axial_velocity_m_s,
        velocity,
        atol=1e-12,
    )
    assert motion.reconstruction_residual_m_s < 1e-12


def test_finite_axis_is_invariant_to_reference_point_choice() -> None:
    omega = np.array([1.0, 2.0, -3.0])
    first_point = np.array([0.2, -0.5, 1.1])
    first_velocity = np.array([4.0, -2.0, 3.0])
    second_point = np.array([-0.3, 0.4, 0.7])
    second_velocity = first_velocity + np.cross(omega, second_point - first_point)

    first = analyze_twist(np.concatenate([omega, first_velocity]), first_point)
    second = analyze_twist(np.concatenate([omega, second_velocity]), second_point)

    np.testing.assert_allclose(first.axis_direction, second.axis_direction, atol=1e-12)
    delta = second.axis_point_m - first.axis_point_m
    assert np.linalg.norm(np.cross(delta, first.axis_direction)) < 1e-12
    assert first.pitch_m_rad == pytest.approx(second.pitch_m_rad, abs=1e-12)


@pytest.mark.parametrize(
    ("twist", "kind"),
    [
        (np.array([0.0, 0.0, 0.0, 3.0, 4.0, 0.0]), MotionKind.TRANSLATION),
        (np.zeros(6), MotionKind.STATIONARY),
    ],
)
def test_degenerate_motion_never_invents_a_finite_axis(
    twist: np.ndarray, kind: MotionKind
) -> None:
    motion = analyze_twist(twist, np.array([1.0, 2.0, 3.0]))

    assert motion.kind is kind
    assert motion.axis_point_m is None
    assert motion.radius_m is None
    assert motion.pitch_m_rad is None


def test_projection_breakdown_preserves_signed_velocity_components() -> None:
    motion = analyze_twist(
        np.array([0.0, 0.0, 2.0, 4.0, -3.0, 1.5]),
        np.zeros(3),
    )

    projections = project_motion(motion)

    assert projections["target"].total_m_s == pytest.approx(4.0)
    assert projections["vertical"].total_m_s == pytest.approx(-3.0)
    assert projections["lateral"].total_m_s == pytest.approx(1.5)
    for projection in projections.values():
        assert projection.total_m_s == pytest.approx(
            projection.orbital_m_s + projection.axial_m_s
        )


def test_joint_contributions_reconstruct_planar_two_link_endpoint_velocity() -> None:
    times = np.linspace(0.0, 0.02, 21)
    first_rate = 2.0
    second_rate = -1.25
    length_1, length_2 = 0.8, 1.1
    points = []
    for time_s in times:
        first_angle = first_rate * time_s
        second_angle = second_rate * time_s
        shoulder = np.zeros(3)
        wrist = shoulder + length_1 * np.array(
            [math.cos(first_angle), math.sin(first_angle), 0.0]
        )
        head = wrist + length_2 * np.array(
            [math.cos(second_angle), math.sin(second_angle), 0.0]
        )
        points.append(np.vstack([shoulder, wrist, head]))

    joint_motion = analyze_joint_motion(
        times,
        np.stack(points),
        ("joint.shoulder", "joint.wrist"),
    )
    middle = len(times) // 2
    expected = np.cross(
        [0.0, 0.0, first_rate], points[middle][1] - points[middle][0]
    ) + np.cross([0.0, 0.0, second_rate], points[middle][2] - points[middle][1])

    np.testing.assert_allclose(
        joint_motion.endpoint_velocity_m_s[middle], expected, atol=2e-5
    )
    np.testing.assert_allclose(
        joint_motion.contribution_velocity_m_s[middle].sum(axis=0),
        expected,
        atol=2e-5,
    )
    assert joint_motion.reconstruction_residual_m_s[middle] < 2e-5


def test_screw_glyph_is_bounded_and_encodes_handedness() -> None:
    positive = analyze_twist(np.array([0.0, 0.0, 20.0, 5.0, 0.0, 2.0]), np.zeros(3))
    negative = analyze_twist(np.array([0.0, 0.0, -20.0, 5.0, 0.0, 2.0]), np.zeros(3))

    positive_glyph = build_screw_glyph(positive, scene_extent_m=2.0)
    negative_glyph = build_screw_glyph(negative, scene_extent_m=2.0)

    assert positive_glyph is not None
    assert negative_glyph is not None
    assert positive_glyph.axis_line_m.shape == (2, 3)
    assert positive_glyph.helix_m.shape == (96, 3)
    assert np.max(np.linalg.norm(positive_glyph.helix_m, axis=1)) < 5.0
    assert positive_glyph.handedness == 1
    assert negative_glyph.handedness == -1


@pytest.mark.parametrize("club_name", club_names())
def test_every_club_and_wedge_produces_finite_club_screw_motion(
    club_name: str,
) -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=100.0),
            club=get_club(club_name),
        )
    )
    assert run.impact_time_s is not None
    index = int(np.argmin(np.abs(run.swing_times - run.impact_time_s)))
    motion = analyze_twist(run.swing_twists[index], run.swing_positions[index])

    assert motion.kind is MotionKind.FINITE
    assert motion.angular_rate_rad_s > 0.0
    assert motion.reconstruction_residual_m_s < 1e-9
