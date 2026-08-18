"""Deterministic fixtures for post-ground spatial-target projection tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from shared.python.swing_sim.flight import (
    FlightGroundTransferError,
    FlightRegionalGroundPipelineResult,
    execute_regional_ground_from_flight,
)
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
    RegionalGroundExecutionOptions,
    SkidRollSettings,
)
from shared.python.swing_sim.ground.regional_plan_records import (
    regional_plan_request_sha256,
)
from shared.python.swing_sim.solver import (
    BoxTolerance,
    SpatialTarget,
    SurfaceCircleTolerance,
    TargetPoint,
)


def complete_pipeline() -> FlightRegionalGroundPipelineResult:
    """Return deterministic complete-rest flight-through-ground evidence."""
    return execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        _plan(),
        capture_speed_m_s=3.0,
    )


def partial_pipeline() -> FlightRegionalGroundPipelineResult:
    """Return deterministic time-censored regional ground evidence."""
    settings = _settings(max_time_s=0.35)
    return execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        settings,
        _plan(settings),
        capture_speed_m_s=3.0,
    )


def left_surface_pipeline() -> FlightRegionalGroundPipelineResult:
    """Return deterministic complete-but-not-rest regional evidence."""
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


def failed_pipeline() -> FlightRegionalGroundPipelineResult:
    """Return deterministic regional execution failure evidence."""
    return execute_regional_ground_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        _plan(),
        capture_speed_m_s=3.0,
        options=RegionalGroundExecutionOptions(settings=SkidRollSettings(max_steps=1)),
    )


def nonsettled_pipeline(
    reason: BounceTerminationReason,
) -> FlightRegionalGroundPipelineResult:
    """Return one exact non-settled bounce outcome with no regional evidence."""
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


def transfer_failure() -> FlightGroundTransferError:
    """Return the exact typed no-contact transfer failure fixture."""
    with pytest.raises(FlightGroundTransferError) as captured:
        execute_regional_ground_from_flight(
            _no_contact_result(),
            _launch(),
            _settings(),
            _plan(),
        )
    return captured.value


def landing_target(
    x_m: float,
    elevation_m: float,
    right_m: float,
    radius_m: float = 0.01,
) -> SpatialTarget:
    """Return an exact course-surface landing target."""
    return SpatialTarget(
        label="Raised green",
        kind="landing_area",
        point=TargetPoint(x_m, elevation_m, right_m),
        tolerance=SurfaceCircleTolerance(radius_m),
        elevation_source="course_surface",
        ground_source="course.surface/raised-green",
    )


def aerial_target() -> SpatialTarget:
    """Return an exact aerial target that cannot use a ground endpoint."""
    return SpatialTarget(
        label="Apex gate",
        kind="aerial_waypoint",
        point=TargetPoint(100.0, 20.0, 0.0),
        tolerance=BoxTolerance((2.0, 2.0, 2.0)),
        elevation_source="absolute",
    )
