"""Rate-to-shared-wedge impact-kinematics adapter contracts."""

from __future__ import annotations

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
