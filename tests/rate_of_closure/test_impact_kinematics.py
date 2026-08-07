"""Rate-to-shared-wedge impact-kinematics adapter contracts."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from rate_of_closure.club import face_center_point, get_club, hosel_point
from rate_of_closure.model import ImpactScenario, solve
from rate_of_closure.simulation import (
    ContactMode,
    SimulationConfig,
    impact_kinematics_for_run,
    run_simulation,
)
from shared.python.golf_club import WedgeKinematicState, analyze_wedge_kinematics

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_documented_generated_wedge_geometry_pins_shaft_aoa_contribution() -> None:
    """Keep the representative-head example distinct from the 20 mm fixture."""
    club = get_club("Pitching Wedge")
    lean_rad = math.radians(15.0)
    lean_rotation = np.array(
        (
            (math.cos(-lean_rad), -math.sin(-lean_rad), 0.0),
            (math.sin(-lean_rad), math.cos(-lean_rad), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    shaft_axis = lean_rotation @ np.array(
        (0.0, math.sin(math.radians(64.0)), -math.cos(math.radians(64.0)))
    )
    contact_offset = lean_rotation @ (
        np.asarray(face_center_point(club)) - np.asarray(hosel_point(club))
    )
    shaft_omega = math.radians(1307.0) * shaft_axis
    shaft_velocity = np.cross(shaft_omega, contact_offset)
    total_velocity = np.array(
        (
            30.0 * 0.44704 * math.cos(math.radians(10.0)),
            -30.0 * 0.44704 * math.sin(math.radians(10.0)),
            0.0,
        )
    )
    state = WedgeKinematicState(
        frame_id="target_ground",
        reference_position_m=(0.0, 0.0, 0.0),
        reference_velocity_mps=tuple(total_velocity - shaft_velocity),
        angular_velocity_rad_s=tuple(shaft_omega),
        shaft_axis_point_m=(0.0, 0.0, 0.0),
        shaft_axis_unit=tuple(shaft_axis),
        contact_point_m=tuple(contact_offset),
        face_normal_unit=(1.0, 0.0, 0.0),
        leading_edge_tangent_unit=(0.0, 0.0, 1.0),
        ground_up_unit=(0.0, 1.0, 0.0),
        arc_tangent_unit=(1.0, 0.0, 0.0),
        arc_tangent_rate_per_s=(0.0, 0.0, 0.0),
    )

    result = analyze_wedge_kinematics(state)

    assert contact_offset == pytest.approx((-0.00353255, -0.02464453, 0.037573))
    assert result.shaft_rotation_velocity_mps == pytest.approx(
        (0.49766011, -0.16405655, -0.06081723), abs=1e-8
    )
    assert result.shaft_vertical_velocity_share == pytest.approx(0.07044590)
    assert result.without_shaft_aoa_deg == pytest.approx(-9.66593875)
    assert result.shaft_counterfactual_aoa_delta_deg == pytest.approx(-0.33406125)


def test_manual_impact_adapter_reconciles_the_existing_point_velocity() -> None:
    scenario = ImpactScenario(
        clubhead_speed_mph=30.0,
        lie_angle_deg=64.0,
        omega_plane_dps=0.0,
        omega_shaft_dps=1307.0,
        com_to_face_mm=20.0,
    )
    run = run_simulation(
        SimulationConfig(
            scenario=scenario,
            club=get_club("Pitching Wedge"),
            impact_time_s=0.03,
        )
    )

    snapshot = impact_kinematics_for_run(run)
    expected = solve(scenario)

    assert snapshot.event_label == "Impact"
    assert snapshot.geometry_basis == "scenario_shaft_line"
    np.testing.assert_allclose(
        snapshot.analysis.contact_velocity_mps,
        expected.point_velocity_mps,
        atol=1e-12,
    )
    assert snapshot.analysis.total_aoa_deg == pytest.approx(expected.aoa_deviation_deg)


def test_articulated_miss_uses_closest_approach_and_measured_shaft_line() -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=get_club("Pitching Wedge"),
            source_kind="double_pendulum",
            swing_duration_s=0.05,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
        )
    )

    snapshot = impact_kinematics_for_run(run)
    sample_index = int(np.argmin(np.abs(run.swing_times - run.inspection_time_s)))
    expected_axis = (
        run.swing_joints[sample_index, -2] - run.swing_positions[sample_index]
    )
    expected_axis /= np.linalg.norm(expected_axis)

    assert snapshot.event_label == "Closest Approach"
    assert snapshot.geometry_basis == "articulated_wrist_to_reference_shaft_line"
    np.testing.assert_allclose(snapshot.state.shaft_axis_unit, expected_axis)
    assert "no shaft-twist degree of freedom" in snapshot.model_limitations


def test_curved_face_leading_edge_is_projected_into_the_tangent_plane() -> None:
    scenario = ImpactScenario(
        clubhead_speed_mph=113.0,
        impact_offset_toe_mm=20.0,
        impact_offset_high_mm=8.0,
    )
    run = run_simulation(
        SimulationConfig(scenario=scenario, club=get_club("Driver 10.5°"))
    )

    snapshot = impact_kinematics_for_run(run)

    assert np.dot(
        snapshot.state.face_normal_unit,
        snapshot.state.leading_edge_tangent_unit,
    ) == pytest.approx(0.0, abs=1e-12)
    assert snapshot.contact_dplane.face_angle_deg != pytest.approx(
        snapshot.face_center_dplane.face_angle_deg
    )


def test_face_center_dplane_uses_rigid_body_point_velocity() -> None:
    scenario = ImpactScenario(
        clubhead_speed_mph=30.0,
        lie_angle_deg=64.0,
        omega_plane_dps=0.0,
        omega_shaft_dps=1307.0,
        com_to_face_mm=20.0,
        impact_offset_toe_mm=12.0,
        impact_offset_high_mm=5.0,
    )
    run = run_simulation(
        SimulationConfig(scenario=scenario, club=get_club("Driver 10.5°"))
    )

    snapshot = impact_kinematics_for_run(run)
    pose = snapshot.state
    reference_velocity = np.asarray(pose.reference_velocity_mps)
    angular_velocity = np.asarray(pose.angular_velocity_rad_s)
    lever = np.asarray(snapshot.face_center_point_m) - np.asarray(
        pose.reference_position_m
    )
    expected = reference_velocity + np.cross(angular_velocity, lever)

    np.testing.assert_allclose(snapshot.face_center_velocity_mps, expected, atol=1e-12)
    np.testing.assert_allclose(
        snapshot.face_center_dplane.travel_direction_unit,
        expected / np.linalg.norm(expected),
        atol=1e-12,
    )
    assert snapshot.face_center_dplane.frame_id == "app_frame:x_target,y_up,z_right"
    assert snapshot.face_center_dplane.spin_loft_3d_deg is not None
    assert snapshot.face_center_dplane.planar_spin_loft_deg is not None


def test_impact_adapter_interpolates_an_off_grid_event_state() -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=get_club("Pitching Wedge"),
            impact_time_s=0.03,
        )
    )
    event_index = 4
    event_time = float(
        0.25 * run.swing_times[event_index] + 0.75 * run.swing_times[event_index + 1]
    )
    poses = np.repeat(np.eye(4)[None, :, :], len(run.swing_times), axis=0)
    poses[:, 0, 3] = np.arange(len(run.swing_times), dtype=float)
    twists = np.zeros_like(run.swing_twists)
    twists[:, 3] = 2.0 * np.arange(len(run.swing_times), dtype=float)
    exact_run = replace(
        run,
        impact_time_s=event_time,
        swing_poses=poses,
        swing_positions=poses[:, :3, 3].copy(),
        swing_twists=twists,
    )

    snapshot = impact_kinematics_for_run(exact_run)

    assert snapshot.event_time_s == pytest.approx(event_time)
    assert snapshot.sample_index == event_index + 1
    assert snapshot.state.reference_position_m[0] == pytest.approx(event_index + 0.75)
    assert snapshot.state.reference_velocity_mps[0] == pytest.approx(
        2.0 * (event_index + 0.75)
    )
