"""Contracts and deterministic pipeline tests for impact solution families."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.python.swing_sim.flight.impact_solution_adapter import (
    CenteredClubDeliveryAdapter,
)
from shared.python.swing_sim.flight.impact_solution_contract import (
    ClubProfileId,
    ImpactSolutionRequest,
    ImpactSolutionResult,
)
from shared.python.swing_sim.flight.impact_solution_solver import (
    solve_impact_solution_families,
)
from shared.python.swing_sim.flight.inverse_contract import (
    DecisionVariable,
    FlightObjective,
    InverseFlightRequest,
    ObjectiveMode,
)
from shared.python.swing_sim.flight.result_contract import FlightMetricId

FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "impact_solution_families_golden_v1.json"
)


def _inverse_request() -> InverseFlightRequest:
    return InverseFlightRequest(
        problem_id="centered-driver-180m",
        variables=(
            DecisionVariable("clubhead_speed_mps", "m/s", 38.0, 50.0, 44.0),
            DecisionVariable("attack_angle_deg", "deg", -2.0, 6.0, 2.0),
            DecisionVariable("dynamic_loft_deg", "deg", 8.0, 16.0, 12.0),
        ),
        objectives=(
            FlightObjective(
                FlightMetricId.CARRY_DISTANCE,
                "m",
                ObjectiveMode.TARGET,
                target_value=180.0,
                tolerance=30.0,
            ),
            FlightObjective(
                FlightMetricId.VERTICAL_LAUNCH_ANGLE,
                "deg",
                ObjectiveMode.TARGET,
                target_value=12.0,
                tolerance=20.0,
            ),
        ),
        max_evaluations=9,
        candidate_count=6,
    )


def _request(
    profile: ClubProfileId = ClubProfileId.CENTERED_DRIVER,
) -> ImpactSolutionRequest:
    return ImpactSolutionRequest(
        inverse_request=_inverse_request(),
        club_profile_id=profile,
        flight_model_id="waterloo_penner",
        family_count=3,
        family_radius=0.2,
        sensitivity_fraction=0.01,
        impact_event_time_s=0.0,
    )


def test_request_is_strict_frame_unit_and_model_explicit() -> None:
    request = _request()
    payload = request.to_dict()

    assert payload["target_frame_id"] == "target_frame:x_downrange,y_up,z_right"
    assert payload["delivery_frame_id"] == "app_frame:x_target,y_up,z_right"
    assert payload["impact_reference_point"] == "ball_center_at_first_contact"
    assert payload["convention_id"] == "app_native"
    assert payload["impact_model_id"] == "rigid_body_centered"
    assert ImpactSolutionRequest.from_dict(payload) == request

    wrong_unit = json.loads(request.to_json())
    wrong_unit["inverse_request"]["variables"][0]["unit"] = "mph"
    with pytest.raises(ValueError, match="canonical unit"):
        ImpactSolutionRequest.from_dict(wrong_unit)

    unsupported = json.loads(request.to_json())
    unsupported["inverse_request"]["variables"][0]["parameter_id"] = "shaft_flex"
    with pytest.raises(ValueError, match="unsupported delivery variable"):
        ImpactSolutionRequest.from_dict(unsupported)

    invalid_domain = json.loads(request.to_json())
    invalid_domain["inverse_request"]["variables"][2]["upper_bound"] = 100.0
    with pytest.raises(ValueError, match="supported range"):
        ImpactSolutionRequest.from_dict(invalid_domain)


@pytest.mark.physics
def test_centered_driver_and_iron_adapters_run_declared_pipeline() -> None:
    for profile in (ClubProfileId.CENTERED_DRIVER, ClubProfileId.CENTERED_IRON):
        adapter = CenteredClubDeliveryAdapter(_request(profile))
        evaluation = adapter.evaluate(
            {
                "clubhead_speed_mps": 44.0,
                "attack_angle_deg": 2.0,
                "dynamic_loft_deg": 12.0,
            }
        )

        assert evaluation.status.value == "complete"
        assert evaluation.reason is None
        assert evaluation.launch_metric(FlightMetricId.BALL_SPEED).value > 0.0
        assert evaluation.flight_metric(FlightMetricId.CARRY_DISTANCE).value > 0.0
        assert evaluation.model_manifest.impact_status.value == "available"
        assert evaluation.model_manifest.flight_status.value == "available"


@pytest.mark.physics
def test_solver_preserves_ranked_families_intervals_and_diagnostics() -> None:
    result = solve_impact_solution_families(
        _request(), CenteredClubDeliveryAdapter(_request())
    )

    assert result.families
    assert [family.rank for family in result.families] == list(
        range(1, len(result.families) + 1)
    )
    assert all(family.members for family in result.families)
    assert all(family.intervals for family in result.families)
    assert all(family.sensitivities for family in result.families)
    assert all(family.launch_residuals for family in result.families)
    assert all(family.flight_residuals for family in result.families)
    assert result.evaluations_attempted == _request().inverse_request.max_evaluations
    assert (
        len(result.rejected_candidates)
        + sum(len(family.members) for family in result.families)
        == result.evaluations_attempted
    )
    assert (
        ImpactSolutionResult.from_dict(json.loads(result.to_json())).to_dict()
        == result.to_dict()
    )


def test_no_impact_and_model_unavailable_are_never_hidden() -> None:
    no_impact_inverse = InverseFlightRequest(
        problem_id="no-impact",
        variables=(
            DecisionVariable("clubhead_speed_mps", "m/s", 44.0, 44.0, 44.0),
            DecisionVariable("attack_angle_deg", "deg", -80.0, -80.0, -80.0),
            DecisionVariable("dynamic_loft_deg", "deg", 80.0, 80.0, 80.0),
        ),
        objectives=(_inverse_request().objectives[0],),
        max_evaluations=1,
        candidate_count=1,
    )
    no_impact_request = ImpactSolutionRequest(
        inverse_request=no_impact_inverse,
        club_profile_id=ClubProfileId.CENTERED_DRIVER,
        flight_model_id="waterloo_penner",
        family_count=1,
        family_radius=0.2,
        sensitivity_fraction=0.01,
        impact_event_time_s=0.0,
    )
    adapter = CenteredClubDeliveryAdapter(no_impact_request)
    no_impact = adapter.evaluate(
        {
            "clubhead_speed_mps": 44.0,
            "attack_angle_deg": -80.0,
            "dynamic_loft_deg": 80.0,
        }
    )
    assert no_impact.status.value == "no_impact"
    assert no_impact.reason == "nonpositive_normal_approach_speed"

    outside_bounds = CenteredClubDeliveryAdapter(_request()).evaluate(
        {
            "clubhead_speed_mps": 100.0,
            "attack_angle_deg": 2.0,
            "dynamic_loft_deg": 12.0,
        }
    )
    assert outside_bounds.status.value == "failed"
    assert outside_bounds.reason == "forward_pipeline_error"

    unavailable_request = ImpactSolutionRequest(
        inverse_request=_inverse_request(),
        club_profile_id=ClubProfileId.CENTERED_DRIVER,
        flight_model_id="not_a_model",
        family_count=1,
        family_radius=0.2,
        sensitivity_fraction=0.01,
        impact_event_time_s=0.0,
    )
    unavailable = CenteredClubDeliveryAdapter(unavailable_request).evaluate(
        {
            "clubhead_speed_mps": 44.0,
            "attack_angle_deg": 2.0,
            "dynamic_loft_deg": 12.0,
        }
    )
    assert unavailable.status.value == "model_unavailable"
    assert unavailable.reason == "unknown_flight_model"


def test_python_contract_matches_shared_typescript_fixture() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    request = ImpactSolutionRequest.from_dict(payload["request"])
    result = ImpactSolutionResult.from_dict(payload["result"])

    assert request.to_dict() == payload["request"]
    assert result.to_dict() == payload["result"]

    duplicate_index = json.loads(json.dumps(payload["result"]))
    duplicate_index["evaluations_attempted"] = 2
    duplicate_index["rejected_candidates"] = [
        {
            "evaluation_index": 0,
            "parameters": payload["result"]["families"][0]["members"][0]["parameters"],
            "reason": "duplicate",
            "status": "complete",
        }
    ]
    with pytest.raises(ValueError, match="exactly once"):
        ImpactSolutionResult.from_dict(duplicate_index)

    boolean_count = json.loads(json.dumps(payload["result"]))
    boolean_count["evaluations_attempted"] = True
    with pytest.raises(ValueError, match="nonnegative integer"):
        ImpactSolutionResult.from_dict(boolean_count)

    negative_representative = json.loads(json.dumps(payload["result"]))
    negative_representative["families"][0]["representative_evaluation_index"] = -1
    with pytest.raises(ValueError, match="identity fields"):
        ImpactSolutionResult.from_dict(negative_representative)
