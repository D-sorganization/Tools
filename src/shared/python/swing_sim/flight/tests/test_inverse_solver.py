"""Contract and behavior tests for the desired-flight inverse solver (#4195)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from shared.python.swing_sim.flight.inverse_contract import (
    DecisionVariable,
    EvaluatedMetric,
    EvaluationStatus,
    FlightObjective,
    InverseFlightRequest,
    InverseFlightResult,
    ObjectiveMode,
    SolverEvaluation,
    SolverStatus,
)
from shared.python.swing_sim.flight.inverse_solver import solve_inverse_flight
from shared.python.swing_sim.flight.result_contract import FlightMetricId

FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "inverse_flight_solver_golden_v1.json"
)


def _request() -> InverseFlightRequest:
    return InverseFlightRequest(
        problem_id="analytic-parity",
        variables=(DecisionVariable("speed", "m/s", 0.0, 10.0, 5.0),),
        objectives=(
            FlightObjective(
                FlightMetricId.CARRY_DISTANCE,
                "m",
                ObjectiveMode.TARGET,
                target_value=75.0,
                tolerance=20.0,
                weight=2.0,
            ),
            FlightObjective(
                FlightMetricId.APEX_HEIGHT,
                "m",
                ObjectiveMode.MAXIMIZE,
                lower_bound=5.0,
                tolerance=10.0,
                weight=0.25,
            ),
        ),
        max_evaluations=5,
        candidate_count=3,
    )


def _analytic_evaluator(parameters: dict[str, float]) -> SolverEvaluation:
    speed = parameters["speed"]
    return SolverEvaluation(
        EvaluationStatus.COMPLETE,
        (
            EvaluatedMetric(
                FlightMetricId.CARRY_DISTANCE, 10.0 * speed, "analytic.carry"
            ),
            EvaluatedMetric(FlightMetricId.APEX_HEIGHT, speed, "analytic.apex"),
        ),
    )


def test_solver_is_deterministic_and_returns_ranked_candidates() -> None:
    result = solve_inverse_flight(_request(), _analytic_evaluator)

    assert result.status is SolverStatus.SOLVED
    assert result.evaluations_completed == 5
    assert len(result.candidates) == 3
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3]
    assert result.candidates[0].parameter("speed").value == pytest.approx(7.5)
    assert result.candidates[0].feasible
    assert (
        result.candidates[0].residual(FlightMetricId.CARRY_DISTANCE).actual_value == 75
    )
    assert result.candidates[1].parameter("speed").value == pytest.approx(6.25)
    assert (
        result.to_json()
        == solve_inverse_flight(_request(), _analytic_evaluator).to_json()
    )


def test_contract_rejects_wrong_units_and_ineligible_metrics() -> None:
    with pytest.raises(ValueError, match="canonical unit"):
        FlightObjective(
            FlightMetricId.CARRY_DISTANCE,
            "yd",
            ObjectiveMode.TARGET,
            target_value=200.0,
        )
    with pytest.raises(ValueError, match="not solver-eligible"):
        FlightObjective(
            FlightMetricId.INITIAL_VELOCITY,
            "m/s",
            ObjectiveMode.TARGET,
            target_value=50.0,
        )
    with pytest.raises(ValueError, match="unique"):
        InverseFlightRequest(
            "duplicate",
            (DecisionVariable("speed", "m/s", 0.0, 10.0, 5.0),),
            (_request().objectives[0], _request().objectives[0]),
            5,
            2,
        )
    with pytest.raises(ValueError, match="positive integer"):
        InverseFlightRequest(
            "fractional-budget",
            (DecisionVariable("speed", "m/s", 0.0, 10.0, 5.0),),
            (_request().objectives[0],),
            2.5,  # type: ignore[arg-type]
            1,
        )


def test_static_infeasibility_and_evaluation_failures_are_typed() -> None:
    impossible = InverseFlightRequest(
        "contradictory-target",
        (DecisionVariable("speed", "m/s", 0.0, 10.0, 5.0),),
        (
            FlightObjective(
                FlightMetricId.CARRY_DISTANCE,
                "m",
                ObjectiveMode.TARGET,
                target_value=100.0,
                lower_bound=0.0,
                upper_bound=50.0,
            ),
        ),
        4,
        2,
    )
    infeasible = solve_inverse_flight(impossible, _analytic_evaluator)
    assert infeasible.status is SolverStatus.INFEASIBLE
    assert infeasible.evaluations_attempted == 0
    assert infeasible.termination_reason == "target_outside_objective_bounds"

    no_impact = solve_inverse_flight(
        _request(),
        lambda _parameters: SolverEvaluation(
            EvaluationStatus.NO_IMPACT, (), "club_did_not_contact_ball"
        ),
    )
    assert no_impact.status is SolverStatus.NO_IMPACT
    assert no_impact.no_impact_count == _request().max_evaluations
    assert not no_impact.candidates

    nonconverged = solve_inverse_flight(
        _request(),
        lambda _parameters: SolverEvaluation(
            EvaluationStatus.NONCONVERGED, (), "integrator_budget_exhausted"
        ),
    )
    assert nonconverged.status is SolverStatus.NONCONVERGED
    assert nonconverged.failed_count == _request().max_evaluations

    malformed = solve_inverse_flight(
        _request(),
        lambda _parameters: object(),  # type: ignore[arg-type,return-value]
    )
    assert malformed.status is SolverStatus.NONCONVERGED
    assert malformed.failed_count == _request().max_evaluations


def test_request_and_result_match_shared_parity_fixture() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    request = InverseFlightRequest.from_dict(fixture["request"])
    result = solve_inverse_flight(request, _analytic_evaluator)

    assert request.to_dict() == fixture["request"]
    assert InverseFlightResult.from_dict(json.loads(result.to_json())) == result
    invalid_result = json.loads(result.to_json())
    invalid_result["candidates"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="solution candidate fields"):
        InverseFlightResult.from_dict(invalid_result)
    digest = hashlib.sha256(result.to_json().encode()).hexdigest()
    assert digest == fixture["result_sha256"]
