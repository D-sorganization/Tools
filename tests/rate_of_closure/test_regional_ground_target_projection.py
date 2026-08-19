"""Qualified post-ground spatial-target projection tests for #4192/#4273."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rate_of_closure.variation.regional_ground_target_projection import (
    MAX_REGIONAL_GROUND_TARGET_ROWS,
    RegionalGroundTargetAvailability,
    build_regional_ground_target_ensemble,
    project_regional_ground_target,
)
from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.ground import (
    BounceTerminationReason,
    GroundFrame,
    GroundResultStatus,
    RegionalGroundExecutionFailureReason,
    RegionalGroundExecutionStatus,
)
from shared.python.swing_sim.solver import (
    SpatialTarget,
    SurfaceCircleTolerance,
    TargetPoint,
)
from tests.rate_of_closure.regional_ground_target_support import (
    aerial_target as _aerial_target,
)
from tests.rate_of_closure.regional_ground_target_support import (
    complete_pipeline as _complete_pipeline,
)
from tests.rate_of_closure.regional_ground_target_support import (
    failed_pipeline as _failed_pipeline,
)
from tests.rate_of_closure.regional_ground_target_support import (
    landing_target as _landing_target,
)
from tests.rate_of_closure.regional_ground_target_support import (
    left_surface_pipeline as _left_surface_pipeline,
)
from tests.rate_of_closure.regional_ground_target_support import (
    nonsettled_pipeline as _nonsettled_pipeline,
)
from tests.rate_of_closure.regional_ground_target_support import (
    partial_pipeline as _partial_pipeline,
)
from tests.rate_of_closure.regional_ground_target_support import (
    transfer_failure as _transfer_failure,
)


def test_complete_rest_projects_once_to_declared_surface_and_reuses_target_miss() -> (
    None
):
    pipeline = _complete_pipeline()
    ground = pipeline.ground_result
    assert ground is not None and ground.summary is not None
    final = ground.trajectory[-1].position_m
    target = _landing_target(final[0] - 2.0, 2.75, final[2] + 3.0)

    projection = project_regional_ground_target(pipeline, target)
    expected = target.miss((final[0], target.point.elevation_m, final[2]))

    assert projection.availability is RegionalGroundTargetAvailability.AVAILABLE
    assert projection.frame is GroundFrame.TARGET
    assert projection.phase == "complete"
    assert projection.reason == "complete_rest"
    assert projection.endpoint_app_m == pytest.approx((final[0], 2.75, final[2]))
    assert projection.miss == expected
    assert projection.hold is False
    assert projection.miss is not None
    assert projection.miss.downrange_m > 0.0
    assert projection.miss.elevation_m == pytest.approx(0.0)
    assert projection.miss.right_m < 0.0
    assert projection.ground_request_sha256 == pipeline.ground_request_sha256
    assert projection.regional_plan_sha256 == pipeline.regional_plan_sha256


def test_complete_rest_hold_populates_target_scalar_ensemble() -> None:
    pipeline = _complete_pipeline()
    ground = pipeline.ground_result
    assert ground is not None
    final = ground.trajectory[-1].position_m
    target = _landing_target(final[0], 8.25, final[2], radius_m=0.5)

    dataset = build_regional_ground_target_ensemble(
        (pipeline,),
        target,
        result_id="target-study",
        source_provenance="pytest/exact-head",
        max_rows=4,
        series_id="driver",
    )
    row = dataset.rows[0]

    assert dataset.provenance.adapter_id == "regional-ground-target/scalar-ensemble/v1"
    assert dataset.provenance.source_provenance == "pytest/exact-head"
    assert row.row_id == "series:driver/trial:0"
    assert row.cohort == "hold"
    assert row.values == {
        "target.hold": 1.0,
        "target.miss_distance": 0.0,
        "target.miss_downrange": 0.0,
        "target.miss_elevation": 0.0,
        "target.miss_lateral": 0.0,
    }
    assert row.attributes is not None
    assert row.attributes["availability"] == "AVAILABLE"
    assert row.attributes["frame"] == "target_frame:x_downrange,y_up,z_right"
    assert row.attributes["phase"] == "complete"
    assert row.attributes["reason"] == "complete_rest"
    assert row.attributes["target_kind"] == "landing_area"
    assert row.attributes["target_ground_source"] == "course.surface/raised-green"


def test_authored_frames_share_one_canonical_endpoint_and_miss() -> None:
    pipeline = _complete_pipeline()
    ground = pipeline.ground_result
    assert ground is not None
    final = ground.trajectory[-1].position_m
    app_target = _landing_target(final[0] - 1.0, 3.5, final[2] + 2.0)
    flight_target = SpatialTarget(
        label=app_target.label,
        kind="landing_area",
        point=TargetPoint.from_frame(
            (app_target.point.x_m, -app_target.point.right_m, 3.5), "flight"
        ),
        tolerance=SurfaceCircleTolerance(0.01),
        elevation_source="course_surface",
        ground_source=app_target.ground_source,
    )

    app_projection = project_regional_ground_target(pipeline, app_target)
    flight_projection = project_regional_ground_target(pipeline, flight_target)

    assert app_target.point.app_coordinates_m == pytest.approx(
        flight_target.point.app_coordinates_m
    )
    assert app_projection.endpoint_app_m == pytest.approx(
        flight_projection.endpoint_app_m
    )
    assert app_projection.miss == flight_projection.miss


def test_aerial_target_is_typed_and_never_flattened_to_ground_endpoint() -> None:
    pipeline = _complete_pipeline()

    projection = project_regional_ground_target(pipeline, _aerial_target())
    row = build_regional_ground_target_ensemble(
        (pipeline,),
        _aerial_target(),
        "aerial-study",
        "pytest/exact-head",
        1,
    ).rows[0]

    assert (
        projection.availability
        is RegionalGroundTargetAvailability.AERIAL_REQUIRES_FLIGHT_TRAJECTORY
    )
    assert projection.phase == "target"
    assert projection.reason == "AERIAL_REQUIRES_FLIGHT_TRAJECTORY"
    assert projection.endpoint_app_m is None
    assert projection.miss is None
    assert projection.hold is None
    assert row.cohort == "unavailable"
    assert all(value is None for value in row.values.values())
    assert row.attributes is not None
    assert row.attributes["availability"] == "AERIAL_REQUIRES_FLIGHT_TRAJECTORY"


@pytest.mark.parametrize(
    ("outcome_factory", "availability", "phase", "reason"),
    [
        (
            _transfer_failure,
            RegionalGroundTargetAvailability.TRANSFER_ERROR,
            "flight_transfer",
            "no_physical_contact",
        ),
        (
            lambda: _nonsettled_pipeline(BounceTerminationReason.TIME_LIMIT),
            RegionalGroundTargetAvailability.BOUNCE_NOT_SETTLED,
            "bounce",
            "time_limit",
        ),
        (
            lambda: _nonsettled_pipeline(BounceTerminationReason.EVENT_LIMIT),
            RegionalGroundTargetAvailability.BOUNCE_NOT_SETTLED,
            "bounce",
            "event_limit",
        ),
        (
            lambda: _nonsettled_pipeline(BounceTerminationReason.NO_RECONTACT),
            RegionalGroundTargetAvailability.BOUNCE_NOT_SETTLED,
            "bounce",
            "no_recontact",
        ),
        (
            lambda: _nonsettled_pipeline(BounceTerminationReason.CANCELLED),
            RegionalGroundTargetAvailability.BOUNCE_NOT_SETTLED,
            "bounce",
            "cancelled",
        ),
        (
            lambda: _nonsettled_pipeline(BounceTerminationReason.NUMERICAL_FAILURE),
            RegionalGroundTargetAvailability.BOUNCE_NOT_SETTLED,
            "bounce",
            "numerical_failure",
        ),
        (
            _failed_pipeline,
            RegionalGroundTargetAvailability.REGIONAL_FAILED,
            "regional_ground",
            "step_limit",
        ),
        (
            _partial_pipeline,
            RegionalGroundTargetAvailability.REGIONAL_INCOMPLETE,
            "regional_ground",
            "partial",
        ),
        (
            _left_surface_pipeline,
            RegionalGroundTargetAvailability.GROUND_NOT_REST,
            "ground",
            "left_surface",
        ),
    ],
)
def test_unqualified_outcomes_are_typed_nulls(
    outcome_factory,
    availability: RegionalGroundTargetAvailability,
    phase: str,
    reason: str,
) -> None:
    outcome = outcome_factory()
    projection = project_regional_ground_target(outcome, _landing_target(10, 0, 0))
    row = build_regional_ground_target_ensemble(
        (outcome,),
        _landing_target(10, 0, 0),
        "unavailable-study",
        "pytest/exact-head",
        1,
    ).rows[0]

    assert projection.availability is availability
    assert projection.phase == phase
    assert projection.reason == reason
    assert projection.endpoint_app_m is None
    assert projection.miss is None
    assert projection.hold is None
    assert row.cohort == "unavailable"
    assert all(value is None for value in row.values.values())
    assert row.attributes is not None
    assert row.attributes["availability"] == availability.value
    assert row.attributes["phase"] == phase
    assert row.attributes["reason"] == reason


def test_regional_cancel_and_missing_summary_fail_closed() -> None:
    complete = _complete_pipeline()
    regional = complete.regional_result
    ground = complete.ground_result
    assert regional is not None and ground is not None
    cancelled = replace(
        complete,
        regional_result=replace(
            regional,
            status=RegionalGroundExecutionStatus.CANCELLED,
            failure_reason=RegionalGroundExecutionFailureReason.CANCELLED,
            ground_result=None,
            transitions=(),
        ),
    )
    object.__setattr__(ground, "summary", None)

    cancelled_projection = project_regional_ground_target(
        cancelled, _landing_target(10, 0, 0)
    )
    missing_projection = project_regional_ground_target(
        complete, _landing_target(10, 0, 0)
    )

    assert (
        cancelled_projection.availability
        is RegionalGroundTargetAvailability.REGIONAL_CANCELLED
    )
    assert cancelled_projection.reason == "cancelled"
    assert (
        missing_projection.availability
        is RegionalGroundTargetAvailability.SUMMARY_UNAVAILABLE
    )
    assert missing_projection.reason == "summary_unavailable"


def test_builder_enforces_exact_types_row_cap_nonempty_identity_and_order() -> None:
    complete = _complete_pipeline()
    target = _landing_target(10.0, 0.0, 0.0)
    dataset = build_regional_ground_target_ensemble(
        (complete, complete), target, "ordered", "pytest", 2
    )
    assert [row.trial_index for row in dataset.rows] == [0, 1]
    assert [row.row_id for row in dataset.rows] == ["trial:0", "trial:1"]
    assert MAX_REGIONAL_GROUND_TARGET_ROWS == 100_000

    with pytest.raises(ContractViolationError, match="max_rows"):
        build_regional_ground_target_ensemble(
            (complete, complete), target, "overflow", "pytest", 1
        )
    with pytest.raises(ContractViolationError, match="nonempty"):
        build_regional_ground_target_ensemble((), target, "empty", "pytest", 1)
    with pytest.raises(ContractViolationError, match="exact SpatialTarget"):
        build_regional_ground_target_ensemble(
            (complete,),
            object(),
            "invalid",
            "pytest",
            1,  # type: ignore[arg-type]
        )
    with pytest.raises(
        ContractViolationError, match="pipeline result or transfer failure"
    ):
        project_regional_ground_target(object(), target)  # type: ignore[arg-type]


def test_partial_positive_summary_is_never_promoted_to_target_numerics() -> None:
    partial = _partial_pipeline()
    ground = partial.ground_result
    assert ground is not None and ground.summary is not None
    assert ground.status is GroundResultStatus.PARTIAL
    assert ground.summary.final_downrange_m > 0.0

    projection = project_regional_ground_target(
        partial,
        _landing_target(ground.summary.final_downrange_m, 0.0, 0.0, radius_m=100.0),
    )

    assert projection.hold is None
    assert projection.miss is None
    assert projection.endpoint_app_m is None
