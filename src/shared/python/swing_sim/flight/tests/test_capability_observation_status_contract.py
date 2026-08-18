"""Adversarial status and metric invariants for capability observations."""

from __future__ import annotations

import pytest

from shared.python.swing_sim.flight.capability_observation import (
    CapabilitySampleMetric,
    CapabilitySampleObservation,
    CapabilitySampleParameter,
    CapabilitySampleStatus,
)
from shared.python.swing_sim.flight.inverse_contract import EvaluationStatus
from shared.python.swing_sim.flight.result_contract import FlightMetricId


def _observation(
    source_status: EvaluationStatus | None,
    effective_status: CapabilitySampleStatus,
    metrics: tuple[CapabilitySampleMetric, ...],
    reason_code: str | None,
) -> CapabilitySampleObservation:
    return CapabilitySampleObservation(
        "status-contract",
        0,
        1,
        1,
        0,
        0,
        0,
        "iron-7",
        (CapabilitySampleParameter("speed", "m/s", 40.0, 41.0),),
        source_status,
        effective_status,
        reason_code,
        reason_code if source_status not in {None, EvaluationStatus.COMPLETE} else None,
        metrics,
    )


def _metric(metric_id: FlightMetricId) -> CapabilitySampleMetric:
    return CapabilitySampleMetric(metric_id, 1.0, "status.fixture")


@pytest.mark.parametrize(
    ("source_status", "effective_status", "reason_code"),
    [
        (EvaluationStatus.NO_IMPACT, CapabilitySampleStatus.NO_IMPACT, "missed_ball"),
        (EvaluationStatus.FAILED, CapabilitySampleStatus.FAILED, "flight_failed"),
        (
            EvaluationStatus.NONCONVERGED,
            CapabilitySampleStatus.FAILED,
            "iteration_limit",
        ),
        (None, CapabilitySampleStatus.FAILED, "invalid_evaluator_result"),
        (None, CapabilitySampleStatus.FAILED, "evaluator_exception"),
    ],
)
def test_noncomplete_statuses_reject_metrics(
    source_status: EvaluationStatus | None,
    effective_status: CapabilitySampleStatus,
    reason_code: str,
) -> None:
    with pytest.raises(ValueError, match="metrics"):
        _observation(
            source_status,
            effective_status,
            (_metric(FlightMetricId.APEX_HEIGHT),),
            reason_code,
        )


@pytest.mark.parametrize(
    "metrics",
    [
        (),
        (_metric(FlightMetricId.CARRY_DISTANCE),),
        (_metric(FlightMetricId.CARRY_OFFLINE),),
        (_metric(FlightMetricId.APEX_HEIGHT),),
    ],
)
def test_complete_status_requires_both_landing_metrics(
    metrics: tuple[CapabilitySampleMetric, ...],
) -> None:
    with pytest.raises(ValueError, match="carry_distance and carry_offline"):
        _observation(
            EvaluationStatus.COMPLETE,
            CapabilitySampleStatus.COMPLETE,
            metrics,
            None,
        )


def test_missing_landing_failure_may_retain_valid_complete_source_metrics() -> None:
    observation = _observation(
        EvaluationStatus.COMPLETE,
        CapabilitySampleStatus.FAILED,
        (_metric(FlightMetricId.APEX_HEIGHT),),
        "missing_required_landing_metrics",
    )
    assert observation.metrics[0].metric_id is FlightMetricId.APEX_HEIGHT

    with pytest.raises(ValueError, match="missing at least one"):
        _observation(
            EvaluationStatus.COMPLETE,
            CapabilitySampleStatus.FAILED,
            (
                _metric(FlightMetricId.CARRY_DISTANCE),
                _metric(FlightMetricId.CARRY_OFFLINE),
            ),
            "missing_required_landing_metrics",
        )
