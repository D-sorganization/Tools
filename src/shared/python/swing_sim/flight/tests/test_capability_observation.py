"""Streaming capability-sample observation and cancellation contracts."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields
from typing import Any

import pytest

from shared.python.swing_sim.flight.capability_contract import (
    CapabilityObjective,
    CapabilityParameter,
    ClubCapability,
    OptimizationRequest,
    PlayerCapabilityProfile,
    TargetDefinition,
)
from shared.python.swing_sim.flight.capability_observation import (
    CAPABILITY_SAMPLE_OBSERVATION_SCHEMA_VERSION,
    CapabilityOptimizationCancelled,
    CapabilityOptimizationHooks,
    CapabilitySampleObservation,
)
from shared.python.swing_sim.flight.capability_optimizer import optimize_capability
from shared.python.swing_sim.flight.inverse_contract import (
    EvaluatedMetric,
    EvaluationStatus,
    SolverEvaluation,
)
from shared.python.swing_sim.flight.result_contract import FlightMetricId

_SOURCE_STATUSES = (
    EvaluationStatus.COMPLETE,
    EvaluationStatus.NO_IMPACT,
    EvaluationStatus.FAILED,
    EvaluationStatus.NONCONVERGED,
    None,
    EvaluationStatus.COMPLETE,
    None,
)
_EFFECTIVE_STATUSES = (
    "complete",
    "no_impact",
    "failed",
    "failed",
    "failed",
    "failed",
    "failed",
)
_REASON_CODES = (
    None,
    "missed_ball",
    "flight_failed",
    "iteration_limit",
    "invalid_evaluator_result",
    "missing_required_landing_metrics",
    "evaluator_exception",
)
_SOURCE_REASONS = (
    None,
    "missed_ball",
    "flight_failed",
    "iteration_limit",
    None,
    None,
    None,
)


def _club(club_id: str = "iron-7") -> ClubCapability:
    parameters = (
        CapabilityParameter("speed", "m/s", 35, 60, 40, 58, 48, 0, 1),
        CapabilityParameter("direction", "deg", -8, 8, -4, 4, 0, 0, 1),
    )
    return ClubCapability(
        club_id,
        parameters,
        "correlation",
        ((1, 0), (0, 1)),
        f"session:{club_id}",
        0.9,
    )


def _profile(*clubs: ClubCapability) -> PlayerCapabilityProfile:
    return PlayerCapabilityProfile("player", clubs or (_club(),), "fit-v1", 0.8)


def _request(
    *club_ids: str, samples: int = 2, candidates: int = 2
) -> OptimizationRequest:
    return OptimizationRequest(
        "problem-17",
        CapabilityObjective.MAXIMIZE_CARRY,
        club_ids or ("iron-7",),
        TargetDefinition("green", 100, 0, 8, 12, 14),
        candidates,
        samples,
        candidates,
        19,
        0.8,
        0.5,
    )


def _complete(_club_id: str, parameters: dict[str, float]) -> SolverEvaluation:
    return SolverEvaluation(
        EvaluationStatus.COMPLETE,
        (
            EvaluatedMetric(
                FlightMetricId.CARRY_OFFLINE,
                parameters["direction"],
                "test.offline",
            ),
            EvaluatedMetric(
                FlightMetricId.CARRY_DISTANCE,
                parameters["speed"] * 2,
                "test.carry",
            ),
        ),
    )


def test_observations_are_ordered_immutable_and_do_not_change_result() -> None:
    profile = _profile(_club("iron-7"), _club("driver"))
    request = _request("iron-7", "driver", samples=2, candidates=3)
    observations: list[CapabilitySampleObservation] = []

    observed = optimize_capability(
        profile,
        request,
        _complete,
        hooks=CapabilityOptimizationHooks(observation_sink=observations.append),
    )
    baseline = optimize_capability(profile, request, _complete)

    assert observed.to_json() == baseline.to_json()
    assert "observations" not in {item.name for item in fields(observed)}
    assert [item.attempt_ordinal for item in observations] == list(range(6))
    assert [item.attempted_count for item in observations] == list(range(1, 7))
    assert [item.total_count for item in observations] == [6] * 6
    assert [item.candidate_ordinal for item in observations] == [0, 0, 1, 1, 2, 2]
    assert [item.club_candidate_ordinal for item in observations] == [0, 0, 0, 0, 1, 1]
    assert [item.sample_ordinal for item in observations] == [0, 1, 0, 1, 0, 1]
    assert [item.club_id for item in observations] == [
        "iron-7",
        "iron-7",
        "driver",
        "driver",
        "iron-7",
        "iron-7",
    ]
    first = observations[0]
    assert first.schema_version == CAPABILITY_SAMPLE_OBSERVATION_SCHEMA_VERSION
    assert [item.parameter_id for item in first.parameters] == ["speed", "direction"]
    assert [item.metric_id for item in first.metrics] == [
        FlightMetricId.CARRY_OFFLINE,
        FlightMetricId.CARRY_DISTANCE,
    ]
    with pytest.raises(FrozenInstanceError):
        first.attempt_ordinal = 3


def test_status_normalization_preserves_safe_source_details() -> None:
    request = _request(samples=7, candidates=1)
    evaluations: list[object] = [
        _complete("iron-7", {"speed": 48, "direction": 0}),
        SolverEvaluation(EvaluationStatus.NO_IMPACT, (), "missed_ball"),
        SolverEvaluation(EvaluationStatus.FAILED, (), "flight_failed"),
        SolverEvaluation(EvaluationStatus.NONCONVERGED, (), "iteration_limit"),
        object(),
        SolverEvaluation(
            EvaluationStatus.COMPLETE,
            (EvaluatedMetric(FlightMetricId.APEX_HEIGHT, 22, "test.apex"),),
        ),
    ]
    observations: list[CapabilitySampleObservation] = []
    calls = 0

    def evaluator(_club_id: str, _parameters: dict[str, float]) -> object:
        nonlocal calls
        index = calls
        calls += 1
        if index == 6:
            raise RuntimeError("sensitive raw exception text")
        return evaluations[index]

    result = optimize_capability(
        _profile(),
        request,
        evaluator,
        hooks=CapabilityOptimizationHooks(observation_sink=observations.append),
    )

    assert tuple(item.source_status for item in observations) == _SOURCE_STATUSES
    assert tuple(item.effective_status for item in observations) == _EFFECTIVE_STATUSES
    assert tuple(item.reason_code for item in observations) == _REASON_CODES
    assert tuple(item.source_reason for item in observations) == _SOURCE_REASONS
    assert result.evaluations_completed == 1
    assert result.no_impact_count == 1
    assert result.failed_count == 5
    assert "sensitive raw exception text" not in str(observations[-1].to_wire())


def test_wire_shape_uses_ordered_lists_and_exact_identity_fields() -> None:
    observations: list[CapabilitySampleObservation] = []
    optimize_capability(
        _profile(),
        _request(samples=1, candidates=1),
        _complete,
        hooks=CapabilityOptimizationHooks(observation_sink=observations.append),
    )

    wire = observations[0].to_wire()
    assert list(wire) == [
        "schema_version",
        "problem_id",
        "attempt_ordinal",
        "attempted_count",
        "total_count",
        "candidate_ordinal",
        "club_candidate_ordinal",
        "sample_ordinal",
        "club_id",
        "parameters",
        "source_status",
        "effective_status",
        "reason_code",
        "source_reason",
        "metrics",
    ]
    assert [item["parameter_id"] for item in wire["parameters"]] == [
        "speed",
        "direction",
    ]
    assert [item["metric_id"] for item in wire["metrics"]] == [
        "carry_offline",
        "carry_distance",
    ]


def test_mutated_nonfinite_evaluation_is_an_invalid_evaluator_result() -> None:
    evaluation = _complete("iron-7", {"speed": 48, "direction": 0})
    object.__setattr__(evaluation.metrics[1], "value", math.nan)
    observations: list[CapabilitySampleObservation] = []

    result = optimize_capability(
        _profile(),
        _request(samples=1, candidates=1),
        lambda _club_id, _parameters: evaluation,
        hooks=CapabilityOptimizationHooks(observation_sink=observations.append),
    )

    assert result.failed_count == 1
    assert observations[0].source_status is None
    assert observations[0].effective_status == "failed"
    assert observations[0].reason_code == "invalid_evaluator_result"
    assert observations[0].metrics == ()


def test_unknown_metric_id_is_an_invalid_evaluator_result() -> None:
    evaluation = _complete("iron-7", {"speed": 48, "direction": 0})
    object.__setattr__(evaluation.metrics[0], "metric_id", "hostile_metric")
    observations: list[CapabilitySampleObservation] = []

    result = optimize_capability(
        _profile(),
        _request(samples=1, candidates=1),
        lambda _club_id, _parameters: evaluation,
        hooks=CapabilityOptimizationHooks(observation_sink=observations.append),
    )

    assert result.failed_count == 1
    assert observations[0].source_status is None
    assert observations[0].effective_status == "failed"
    assert observations[0].reason_code == "invalid_evaluator_result"
    assert observations[0].metrics == ()


def test_cancellation_is_typed_and_stops_before_next_attempt() -> None:
    observations: list[CapabilitySampleObservation] = []
    request = _request(samples=4, candidates=2)
    hooks = CapabilityOptimizationHooks(
        observation_sink=observations.append,
        should_cancel=lambda: len(observations) >= 2,
    )

    with pytest.raises(CapabilityOptimizationCancelled) as caught:
        optimize_capability(_profile(), request, _complete, hooks=hooks)

    assert caught.value.attempted_count == 2
    assert caught.value.total_count == 8
    assert len(observations) == 2


@pytest.mark.parametrize("attempted,total", [(-1, 2), (True, 2), (2, 1), (0, 1.5)])
def test_cancellation_progress_counts_fail_closed(attempted: Any, total: Any) -> None:
    with pytest.raises(ValueError):
        CapabilityOptimizationCancelled(attempted, total)


def test_sink_exceptions_propagate_without_additional_evaluations() -> None:
    class SinkFailure(RuntimeError):
        pass

    calls = 0

    def evaluator(club_id: str, parameters: dict[str, float]) -> SolverEvaluation:
        nonlocal calls
        calls += 1
        return _complete(club_id, parameters)

    def sink(_observation: CapabilitySampleObservation) -> None:
        raise SinkFailure("consumer failed")

    with pytest.raises(SinkFailure, match="consumer failed"):
        optimize_capability(
            _profile(),
            _request(samples=3, candidates=2),
            evaluator,
            hooks=CapabilityOptimizationHooks(observation_sink=sink),
        )

    assert calls == 1


def test_empty_hooks_preserve_legacy_call_behavior() -> None:
    profile = _profile()
    request = _request()

    assert optimize_capability(profile, request, _complete) == optimize_capability(
        profile,
        request,
        _complete,
        hooks=CapabilityOptimizationHooks(),
    )
