"""Capability-aware robust flight optimization contracts and behavior."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from shared.python.swing_sim.flight.capability_contract import (
    CapabilityObjective,
    CapabilityParameter,
    ClubCapability,
    OptimizationRequest,
    OptimizationResult,
    PlayerCapabilityProfile,
    TargetDefinition,
)
from shared.python.swing_sim.flight.capability_optimizer import optimize_capability
from shared.python.swing_sim.flight.inverse_contract import (
    EvaluatedMetric,
    EvaluationStatus,
    SolverEvaluation,
)
from shared.python.swing_sim.flight.result_contract import FlightMetricId

FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "capability_optimizer_golden_v1.json"
)


def _profile() -> PlayerCapabilityProfile:
    parameters = (
        CapabilityParameter("ball_speed", "m/s", 40, 60, 45, 58, 50, 0, 1),
        CapabilityParameter("launch_angle", "deg", 5, 25, 8, 20, 12, 0, 1),
        CapabilityParameter("launch_direction", "deg", -8, 8, -4, 4, 0, 0, 1),
    )
    return PlayerCapabilityProfile(
        "player-1",
        (
            ClubCapability(
                "iron-7",
                parameters,
                "correlation",
                ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                "session:a",
                0.9,
            ),
            ClubCapability(
                "driver",
                parameters,
                "correlation",
                ((1, 0.3, 0), (0.3, 1, 0), (0, 0, 1)),
                "session:b",
                0.8,
            ),
        ),
        "fit-v1",
        0.85,
    )


def _request(objective: CapabilityObjective) -> OptimizationRequest:
    return OptimizationRequest(
        "robust-shot",
        objective,
        ("iron-7", "driver"),
        TargetDefinition("green", 105, 0, 8, 15, 16),
        candidate_budget=10,
        ensemble_size=12,
        alternatives_count=4,
        seed=17,
        cvar_alpha=0.8,
        minimum_success_fraction=0.75,
    )


def _evaluator(club_id: str, parameters: dict[str, float]) -> SolverEvaluation:
    speed = parameters["ball_speed"]
    launch = parameters["launch_angle"]
    direction = parameters["launch_direction"]
    club_bonus = 18 if club_id == "driver" else 0
    carry = speed * 2 + launch * 0.4 + club_bonus
    offline = direction * 1.5
    return SolverEvaluation(
        EvaluationStatus.COMPLETE,
        (
            EvaluatedMetric(FlightMetricId.CARRY_DISTANCE, carry, "analytic.carry"),
            EvaluatedMetric(FlightMetricId.CARRY_OFFLINE, offline, "analytic.offline"),
        ),
    )


@pytest.mark.parametrize("objective", tuple(CapabilityObjective))
def test_all_robust_objectives_are_deterministic(
    objective: CapabilityObjective,
) -> None:
    first = optimize_capability(_profile(), _request(objective), _evaluator)
    second = optimize_capability(_profile(), _request(objective), _evaluator)

    assert first.to_json() == second.to_json()
    assert 1 <= len(first.alternatives) <= 4
    assert [item.rank for item in first.alternatives] == list(
        range(1, len(first.alternatives) + 1)
    )
    best = first.alternatives[0]
    assert best.sample_count == 12
    assert 0 <= best.target_hold_probability <= 1
    assert 0 <= best.failure_fraction <= 1
    assert best.cvar_miss_m >= best.expected_miss_m
    assert isinstance(best.limiting_constraints, tuple)
    if objective is CapabilityObjective.DISTANCE_CONTROL_PARETO:
        assert best.pareto_efficient


def test_variability_and_downside_use_distinct_robust_risk_scores() -> None:
    variability = optimize_capability(
        _profile(), _request(CapabilityObjective.MINIMIZE_VARIABILITY), _evaluator
    ).alternatives[0]
    downside = optimize_capability(
        _profile(), _request(CapabilityObjective.MINIMIZE_DOWNSIDE), _evaluator
    ).alternatives[0]

    assert variability.score == pytest.approx(variability.dispersion_rms_m)
    assert downside.score == pytest.approx(
        downside.cvar_miss_m + downside.downside_carry_m
    )
    assert variability.score != pytest.approx(variability.expected_miss_m)


def test_profile_fails_closed_for_invalid_evidence_and_covariance() -> None:
    with pytest.raises(ValueError, match="evidence"):
        CapabilityParameter("speed", "m/s", 40, 60, 20, 50, 45, 0, 1)
    with pytest.raises(ValueError, match="positive semidefinite"):
        ClubCapability(
            "driver",
            (
                CapabilityParameter("a", "m/s", 0, 10, 0, 10, 5, 0, 1),
                CapabilityParameter("b", "deg", 0, 10, 0, 10, 5, 0, 1),
            ),
            "covariance",
            ((1, 2), (2, 1)),
            "session",
            0.8,
        )


def test_failures_confidence_extrapolation_and_constraints_are_reported() -> None:
    def partial(club_id: str, parameters: dict[str, float]) -> SolverEvaluation:
        if parameters["launch_direction"] > 2:
            return SolverEvaluation(EvaluationStatus.NO_IMPACT, (), "missed_ball")
        return _evaluator(club_id, parameters)

    request = replace(
        _request(CapabilityObjective.MAXIMIZE_TARGET_HOLD), alternatives_count=10
    )
    result = optimize_capability(_profile(), request, partial)

    assert result.evaluations_attempted > result.evaluations_completed
    assert any(item.failure_fraction > 0 for item in result.alternatives)
    assert all(0 <= item.confidence <= 1 for item in result.alternatives)
    assert any(item.extrapolated for item in result.alternatives)
    assert any(item.limiting_constraints for item in result.alternatives)

    malformed = optimize_capability(
        _profile(),
        request,
        lambda _club_id, _parameters: object(),  # type: ignore[arg-type,return-value]
    )
    assert malformed.status == "nonconverged"
    assert malformed.failed_count == request.candidate_budget * request.ensemble_size


def test_shared_fixture_round_trip_and_parity() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    profile = PlayerCapabilityProfile.from_dict(fixture["profile"])
    request = OptimizationRequest.from_dict(fixture["request"])
    result = optimize_capability(profile, request, _evaluator)

    assert profile.to_dict() == fixture["profile"]
    assert request.to_dict() == fixture["request"]
    assert OptimizationResult.from_dict(result.to_dict()) == result
    invalid_result = result.to_dict()
    invalid_result["alternatives"][0]["extrapolated"] = "false"  # type: ignore[index]
    with pytest.raises(ValueError, match="boolean"):
        OptimizationResult.from_dict(invalid_result)
    expected = fixture["expected"]
    assert result.alternatives[0].club_id == expected["best_club_id"]
    assert result.alternatives[0].mean_carry_m == pytest.approx(
        expected["mean_carry_m"]
    )
    assert result.alternatives[0].target_hold_probability == pytest.approx(
        expected["target_hold_probability"]
    )
    for objective in (
        CapabilityObjective.MINIMIZE_VARIABILITY,
        CapabilityObjective.MINIMIZE_DOWNSIDE,
    ):
        objective_result = optimize_capability(
            profile, replace(request, objective=objective), _evaluator
        ).alternatives[0]
        objective_expected = fixture["objective_expectations"][objective.value]
        assert objective_result.club_id == objective_expected["best_club_id"]
        assert objective_result.score == pytest.approx(objective_expected["score"])
        if objective is CapabilityObjective.MINIMIZE_VARIABILITY:
            assert objective_result.dispersion_rms_m == pytest.approx(
                objective_expected["dispersion_rms_m"]
            )
        else:
            assert objective_result.cvar_miss_m == pytest.approx(
                objective_expected["cvar_miss_m"]
            )
            assert objective_result.downside_carry_m == pytest.approx(
                objective_expected["downside_carry_m"]
            )
