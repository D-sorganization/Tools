"""Rate-to-shared-wedge ground-clearance adapter contracts."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.club import get_club
from rate_of_closure.model import ImpactScenario, impact_lever_m
from rate_of_closure.simulation import (
    ContactMode,
    SimulationConfig,
    ground_clearance_for_run,
    representative_wedge_parameters_for_club,
    run_simulation,
)
from rate_of_closure.simulation.ground_clearance import _registered_wedge_sweep
from shared.python.golf_club import (
    ContactSequence,
    GroundPlane,
    WedgePreset,
    wedge_face_contact_point_m,
    wedge_preset,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_adapter_uses_actual_ball_contact_for_a_hit() -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=get_club("Pitching Wedge"),
            impact_time_s=0.03,
        )
    )

    snapshot = ground_clearance_for_run(
        run,
        wedge_preset(WedgePreset.MID_BOUNCE),
        GroundPlane(point_m=(0.0, -10.0, 0.0)),
    )
    analysis = snapshot.analysis

    assert analysis.ball_contact_time_s == pytest.approx(0.03)
    assert analysis.sequence is ContactSequence.BALL_ONLY
    assert analysis.first_ground_contact is None
    assert snapshot.geometry_basis == "canonical_wedge_face_contact_registration"
    assert run.impact_outcome.geometry_model in snapshot.model_limitations


def test_adapter_preserves_a_miss_without_fabricating_ball_contact() -> None:
    run = run_simulation(
        SimulationConfig(
            scenario=ImpactScenario(clubhead_speed_mph=30.0),
            club=get_club("Pitching Wedge"),
            source_kind="double_pendulum",
            swing_duration_s=0.05,
            contact_mode=ContactMode.FIXED_BALL_CONTACT,
        )
    )

    snapshot = ground_clearance_for_run(
        run,
        wedge_preset(WedgePreset.LOW_BOUNCE),
        GroundPlane(point_m=(0.0, -10.0, 0.0)),
    )
    analysis = snapshot.analysis

    assert run.impact_time_s is None
    assert analysis.ball_contact_time_s is None
    assert analysis.sequence is ContactSequence.NO_CONTACT_MISS


def test_registration_aligns_face_point_and_shifts_twist_reference() -> None:
    scenario = ImpactScenario(
        clubhead_speed_mph=30.0,
        impact_offset_toe_mm=10.0,
        impact_offset_high_mm=5.0,
    )
    run = run_simulation(
        SimulationConfig(
            scenario=scenario,
            club=get_club("Sand Wedge"),
            impact_time_s=0.03,
        )
    )
    parameters = wedge_preset(WedgePreset.HIGH_BOUNCE)

    poses, twists = _registered_wedge_sweep(run, parameters)
    index = int(np.argmin(np.abs(run.swing_times - run.inspection_time_s)))
    rotation = run.swing_poses[index, :3, :3]
    canonical_contact = np.asarray(wedge_face_contact_point_m(parameters, 0.010, 0.005))
    expected_contact = run.swing_poses[index, :3, 3] + rotation @ impact_lever_m(
        scenario
    )
    registered_contact = poses[index, :3, 3] + rotation @ canonical_contact
    reference_shift = poses[index, :3, 3] - run.swing_poses[index, :3, 3]
    expected_linear_velocity = run.swing_twists[index, 3:] + np.cross(
        run.swing_twists[index, :3], reference_shift
    )

    np.testing.assert_allclose(registered_contact, expected_contact, atol=1e-12)
    np.testing.assert_allclose(twists[index, 3:], expected_linear_velocity, atol=1e-12)


def test_adapter_rejects_non_run_inputs() -> None:
    with pytest.raises(TypeError, match="SimulationRun"):
        ground_clearance_for_run(  # type: ignore[arg-type]
            object(),
            wedge_preset(WedgePreset.MID_BOUNCE),
            GroundPlane(),
        )


def test_representative_parameters_preserve_selected_wedge_static_datums() -> None:
    club = get_club("Sand Wedge")

    parameters = representative_wedge_parameters_for_club(club)

    assert parameters is not None
    assert parameters.loft_deg == pytest.approx(club.loft_deg)
    assert parameters.lie_deg == pytest.approx(club.lie_deg)
    assert parameters.target_mass_kg == pytest.approx(club.head_mass_kg)
    assert parameters.bounce_deg == pytest.approx(10.0)
    assert "illustrative" in parameters.provenance.uncertainty_note.lower()


def test_representative_parameters_are_unavailable_for_non_wedges() -> None:
    assert representative_wedge_parameters_for_club(get_club("Driver 10.5°")) is None
