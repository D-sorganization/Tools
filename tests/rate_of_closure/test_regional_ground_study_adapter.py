"""Qualified regional-ground evidence adapter tests for issue #4273."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rate_of_closure.variation.regional_ground_study_adapter import (
    apply_regional_ground_metrics,
    build_regional_ground_study_ensemble,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.flight import (
    FlightGroundTransferError,
    FlightMetricId,
    FlightMetricInputs,
    FlightRegionalGroundPipelineResult,
    MetricTrajectoryPoint,
    ValueStatus,
    derive_flight_metric_result,
    execute_regional_ground_from_flight,
)
from shared.python.swing_sim.flight.result_metrics import FlightRunManifest
from shared.python.swing_sim.flight.tests._regional_ground_pipeline_support import (
    _crossing_result,
    _empty_termination_pair,
    _launch,
    _no_contact_result,
    _plan,
    _settings,
    _time_limit_pair,
)
from shared.python.swing_sim.ground import (
    BounceTerminationReason,
    GroundResultStatus,
    RegionalGroundExecutionFailureReason,
    RegionalGroundExecutionOptions,
    RegionalGroundExecutionStatus,
    SkidRollSettings,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    regional_plan_request_sha256,
)


def _complete_pipeline() -> FlightRegionalGroundPipelineResult:
    return execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        _plan(),
        capture_speed_m_s=3.0,
    )


def _partial_pipeline() -> FlightRegionalGroundPipelineResult:
    settings = _settings(max_time_s=0.35)
    return execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        settings,
        _plan(settings),
        capture_speed_m_s=3.0,
    )


def _failed_pipeline() -> FlightRegionalGroundPipelineResult:
    return execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        _plan(),
        capture_speed_m_s=3.0,
        options=RegionalGroundExecutionOptions(settings=SkidRollSettings(max_steps=1)),
    )


def _left_surface_pipeline() -> FlightRegionalGroundPipelineResult:
    plan = _plan()
    region = replace(plan.regions[0], lower_coordinate_m=3.5, upper_coordinate_m=3.8)
    bounded = replace(plan, upper_coordinate_m=4.0, regions=(region,))
    return execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        bounded,
        capture_speed_m_s=3.0,
    )


def _nonsettled_pipeline(
    reason: BounceTerminationReason,
) -> FlightRegionalGroundPipelineResult:
    pair = (
        _time_limit_pair()
        if reason is BounceTerminationReason.TIME_LIMIT
        else _empty_termination_pair(reason)
    )
    plan = _plan()
    return FlightRegionalGroundPipelineResult(
        bounce_result=pair,
        regional_plan=plan,
        ground_request_sha256=pair.request.ground_request_sha256,
        repeated_bounce_execution_input_sha256=pair.request.execution_input_sha256,
        regional_plan_sha256=regional_plan_request_sha256(plan),
        regional_result=None,
    )


def _transfer_failure() -> FlightGroundTransferError:
    with pytest.raises(FlightGroundTransferError) as captured:
        execute_regional_ground_from_flight(
            _no_contact_result(),
            _launch(),
            _settings(),
            _plan(),
        )
    return captured.value


def _metric_inputs() -> FlightMetricInputs:
    return FlightMetricInputs(
        trajectory=(
            MetricTrajectoryPoint(0.0, (0.0, 0.0, 0.0), (10.0, 2.0, 0.0)),
            MetricTrajectoryPoint(1.0, (10.0, 1.0, 1.0), (10.0, -1.0, 1.0)),
            MetricTrajectoryPoint(2.0, (20.0, -1.0, 2.0), (10.0, -2.0, 1.0)),
        ),
        spin_vector_rpm=(0.0, 1000.0, 2000.0),
    )


def _manifest() -> FlightRunManifest:
    return FlightRunManifest(
        model_id="fixture-flight",
        model_version="1.0.0",
        integration_status="complete",
        termination_reason="ground_crossing",
        environment=(("air_density_kg_m3", "1.225"),),
        wind=(("wind_x_m_s", "0"),),
        uncertainty_status="deterministic",
    )


def test_complete_rest_populates_existing_metrics_and_distinct_ground_scalars() -> None:
    pipeline = _complete_pipeline()
    ground = pipeline.ground_result
    assert ground is not None and ground.summary is not None

    inputs = apply_regional_ground_metrics(_metric_inputs(), pipeline)
    metrics = derive_flight_metric_result(inputs, _manifest())
    dataset = build_regional_ground_study_ensemble(
        (pipeline,), "study-complete", "pytest/exact-head", 8, series_id="driver"
    )
    row = dataset.rows[0]

    assert metrics.scalar(FlightMetricId.TOTAL_DISTANCE) == pytest.approx(
        ground.summary.total_distance_m
    )
    assert metrics.scalar(FlightMetricId.ROLL_DISTANCE) == pytest.approx(
        ground.summary.roll_distance_m
    )
    assert metrics.scalar(FlightMetricId.BOUNCE_COUNT) == ground.summary.bounce_count
    assert metrics.scalar(FlightMetricId.FINAL_OFFLINE) == pytest.approx(
        ground.summary.final_offline_m
    )
    assert metrics.scalar(FlightMetricId.CARRY_DISTANCE) != pytest.approx(
        metrics.scalar(FlightMetricId.TOTAL_DISTANCE)
    )
    assert row.cohort == "complete"
    assert row.values["metric.total_distance"] == ground.summary.total_distance_m
    assert row.values["ground.bounce_air_distance"] == (
        ground.summary.bounce_air_distance_m
    )
    assert row.values["ground.skid_distance"] == ground.summary.skid_distance_m
    assert row.values["metric.roll_distance"] == ground.summary.roll_distance_m
    assert row.values["metric.final_offline"] == ground.summary.final_offline_m
    assert row.values["metric.bounce_count"] == ground.summary.bounce_count
    assert row.values["metric.carry_distance"] == ground.summary.carry_distance_m
    assert row.values["metric.carry_distance"] != row.values["metric.total_distance"]
    assert row.attributes is not None
    assert row.attributes["endpoint_qualification"] == "complete_rest"
    assert row.attributes["ground_status"] == "complete"
    assert row.attributes["ground_termination"] == "rest"


def test_partial_endpoint_is_typed_but_never_numeric_or_optimizer_eligible() -> None:
    pipeline = _partial_pipeline()
    ground = pipeline.ground_result
    assert ground is not None and ground.summary is not None
    assert ground.status is GroundResultStatus.PARTIAL
    assert ground.summary.total_distance_m > 0.0

    inputs = apply_regional_ground_metrics(_metric_inputs(), pipeline)
    metrics = derive_flight_metric_result(inputs, _manifest())
    dataset = build_regional_ground_study_ensemble(
        (pipeline,), "study-partial", "pytest/exact-head", 8
    )
    row = dataset.rows[0]

    assert inputs.ground_result is None
    assert (
        metrics.value(FlightMetricId.TOTAL_DISTANCE).status is ValueStatus.UNAVAILABLE
    )
    assert row.cohort == "partial"
    assert all(value is None for value in row.values.values())
    assert row.attributes is not None
    assert row.attributes["endpoint_qualification"] == "censored"
    assert row.attributes["regional_status"] == "partial"
    assert row.attributes["ground_status"] == "partial"
    assert row.attributes["ground_termination"] == "time_limit"


def test_complete_left_surface_is_censored_not_final_rest_distance() -> None:
    pipeline = _left_surface_pipeline()
    ground = pipeline.ground_result
    assert ground is not None and ground.summary is not None
    assert ground.status is GroundResultStatus.COMPLETE
    assert ground.termination.reason.value == "left_surface"
    assert ground.summary.total_distance_m > 0.0

    inputs = apply_regional_ground_metrics(_metric_inputs(), pipeline)
    row = build_regional_ground_study_ensemble(
        (pipeline,), "left-surface", "pytest/exact-head", 1
    ).rows[0]

    assert inputs.ground_result is None
    assert row.cohort == "partial"
    assert all(value is None for value in row.values.values())
    assert row.attributes is not None
    assert row.attributes["endpoint_qualification"] == "censored"
    assert row.attributes["ground_status"] == "complete"
    assert row.attributes["ground_termination"] == "left_surface"


def test_failed_cancelled_nonsettled_and_transfer_failures_remain_typed_nulls() -> None:
    failed = _failed_pipeline()
    cancelled = _nonsettled_pipeline(BounceTerminationReason.CANCELLED)
    time_limited = _nonsettled_pipeline(BounceTerminationReason.TIME_LIMIT)
    event_limited = _nonsettled_pipeline(BounceTerminationReason.EVENT_LIMIT)
    no_recontact = _nonsettled_pipeline(BounceTerminationReason.NO_RECONTACT)
    numerical = _nonsettled_pipeline(BounceTerminationReason.NUMERICAL_FAILURE)
    dataset = build_regional_ground_study_ensemble(
        (
            failed,
            cancelled,
            time_limited,
            event_limited,
            no_recontact,
            numerical,
            _transfer_failure(),
        ),
        "study-failures",
        "pytest/exact-head",
        8,
    )

    assert [row.cohort for row in dataset.rows] == [
        "failed",
        "cancelled",
        "partial",
        "partial",
        "partial",
        "failed",
        "unavailable",
    ]
    assert all(
        all(value is None for value in row.values.values()) for row in dataset.rows
    )
    (
        failed_row,
        cancelled_row,
        time_row,
        event_row,
        no_recontact_row,
        numerical_row,
        transfer_row,
    ) = dataset.rows
    assert failed_row.attributes is not None
    assert failed_row.attributes["regional_failure_reason"] == "step_limit"
    assert failed_row.attributes["endpoint_qualification"] == "failed"
    assert cancelled_row.attributes is not None
    assert cancelled_row.attributes["bounce_termination"] == "cancelled"
    assert time_row.attributes is not None
    assert time_row.attributes["bounce_termination"] == "time_limit"
    assert event_row.attributes is not None
    assert event_row.attributes["bounce_termination"] == "event_limit"
    assert no_recontact_row.attributes is not None
    assert no_recontact_row.attributes["bounce_termination"] == "no_recontact"
    assert numerical_row.attributes is not None
    assert numerical_row.attributes["bounce_termination"] == "numerical_failure"
    assert transfer_row.attributes is not None
    assert transfer_row.attributes["source_kind"] == "transfer_failure"
    assert transfer_row.attributes["transfer_field_id"] is not None
    assert transfer_row.attributes["transfer_reason"] is not None


def test_regional_cancelled_result_is_not_treated_as_complete() -> None:
    complete = _complete_pipeline()
    regional = complete.regional_result
    assert regional is not None
    cancelled_regional = replace(
        regional,
        status=RegionalGroundExecutionStatus.CANCELLED,
        failure_reason=RegionalGroundExecutionFailureReason.CANCELLED,
        ground_result=None,
        transitions=(),
    )
    cancelled = replace(complete, regional_result=cancelled_regional)

    row = build_regional_ground_study_ensemble(
        (cancelled,), "regional-cancelled", "pytest/exact-head", 1
    ).rows[0]

    assert row.cohort == "cancelled"
    assert all(value is None for value in row.values.values())
    assert row.attributes is not None
    assert row.attributes["regional_failure_reason"] == "cancelled"


def test_missing_summary_fails_closed_and_clears_stale_metric_input() -> None:
    pipeline = _complete_pipeline()
    ground = pipeline.ground_result
    assert ground is not None
    previously_qualified = apply_regional_ground_metrics(_metric_inputs(), pipeline)
    object.__setattr__(ground, "summary", None)

    cleared = apply_regional_ground_metrics(previously_qualified, pipeline)
    row = build_regional_ground_study_ensemble(
        (pipeline,), "missing-summary", "pytest/exact-head", 1
    ).rows[0]

    assert cleared.ground_result is None
    assert row.cohort == "unavailable"
    assert all(value is None for value in row.values.values())
    assert row.attributes is not None
    assert row.attributes["endpoint_qualification"] == "summary_unavailable"


def test_builder_rejects_overflow_empty_sources_and_non_contract_outcomes() -> None:
    complete = _complete_pipeline()
    with pytest.raises(ContractViolationError, match="nonempty"):
        build_regional_ground_study_ensemble((), "empty", "pytest", 1)
    with pytest.raises(ContractViolationError, match="max_rows"):
        build_regional_ground_study_ensemble(
            (complete, complete), "overflow", "pytest", 1
        )
    with pytest.raises(
        ContractViolationError, match="pipeline result or transfer failure"
    ):
        build_regional_ground_study_ensemble(
            (object(),),  # type: ignore[arg-type]
            "invalid",
            "pytest",
            1,
        )
