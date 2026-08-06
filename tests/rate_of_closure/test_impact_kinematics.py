"""Rate-to-shared-wedge impact-kinematics adapter contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario, solve
from rate_of_closure.simulation import (
    ContactMode,
    SimulationConfig,
    impact_kinematics_for_run,
    run_simulation,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


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
