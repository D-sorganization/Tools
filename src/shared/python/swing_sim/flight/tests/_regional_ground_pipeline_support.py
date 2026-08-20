"""Deterministic fixtures for regional-ground pipeline contract tests."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from shared.python.swing_sim.flight import (
    FlightGroundTransferSettings,
    FlightResult,
    FlightStatePoint,
    LaunchConditions,
    execute_repeated_bounce_from_flight,
    launch_relative_surface,
)
from shared.python.swing_sim.ground import (
    BounceTermination,
    BounceTerminationReason,
    CalibrationKind,
    GroundCalibration,
    GroundFrame,
    GroundProvenance,
    GroundRegionalMaterialPlanRequest,
    GroundRegionalMaterialRegion,
    GroundSurfaceProfile,
    RepeatedBounceRequestResultPair,
    RepeatedBounceResult,
)

BALL_RADIUS_M = 0.02135


def _surface() -> GroundSurfaceProfile:
    return GroundSurfaceProfile(
        surface_id="pipeline-plane",
        provider_id="tools.pipeline-test",
        provider_version="1.0.0",
        frame=GroundFrame.TARGET,
        height_m=0.0,
        normal_unit=(0.0, 1.0, 0.0),
        surface_velocity_m_s=(0.0, 0.0, 0.0),
        normal_restitution=0.4,
        static_friction=0.35,
        kinetic_friction=0.25,
        rolling_resistance=0.04,
        firmness_pa=1_000_000.0,
        hardness_fraction=0.7,
        grass_height_m=0.01,
        compressibility_fraction=0.2,
        compression_damping_fraction=0.2,
        turf_density_kg_m3=180.0,
        moisture_fraction=0.3,
    )


def _settings(
    *, max_time_s: float = 12.0, output_interval_s: float = 0.01
) -> FlightGroundTransferSettings:
    return FlightGroundTransferSettings(
        request_id="flight-regional-pipeline-001",
        surface=_surface(),
        calibration=GroundCalibration(
            "pipeline-calibration", CalibrationKind.MEASURED, "test evidence", 1.0
        ),
        provenance=GroundProvenance("pytest", "1.0", "local", "a" * 64),
        max_time_s=max_time_s,
        output_interval_s=output_interval_s,
        max_events=32,
    )


def _state(
    time_s: float,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
) -> FlightStatePoint:
    return FlightStatePoint(
        time_s,
        np.array(position),
        np.array(velocity),
        np.zeros(3),
    )


def _crossing_result() -> FlightResult:
    return FlightResult(
        (
            _state(0.0, (0.0, 0.0, 0.0), (10.0, 0.0, 5.0)),
            _state(0.1, (1.0, -1.0, 0.05), (10.0, -1.0, 1.0)),
            _state(0.2, (2.0, -2.0, 0.03), (10.0, -1.0, -2.0)),
            _state(0.3, (3.0, -3.0, -0.001), (10.0, -1.0, -2.0)),
        ),
        "pipeline-synthetic",
    )


def _no_contact_result() -> FlightResult:
    return FlightResult(
        (
            _state(0.0, (0.0, 0.0, 0.0), (10.0, 0.0, 5.0)),
            _state(0.1, (1.0, 0.0, 0.05), (10.0, 0.0, 2.0)),
        ),
        "pipeline-no-contact",
    )


def _launch() -> LaunchConditions:
    return LaunchConditions(
        ball_speed=10.0,
        launch_angle=math.radians(20.0),
        spin_rate=0.0,
        ball_radius=BALL_RADIUS_M,
    )


def _plan(
    transfer: FlightGroundTransferSettings | None = None,
) -> GroundRegionalMaterialPlanRequest:
    settings = _settings() if transfer is None else transfer
    launch = _launch()
    base = launch_relative_surface(
        settings.surface,
        launch.ball_radius,
        launch.ball_setup,
    )
    region_surface = replace(
        base,
        surface_id="pipeline-far-region",
        rolling_resistance=0.2,
    )
    return GroundRegionalMaterialPlanRequest(
        request_id="pipeline-plan-001",
        base_surface=base,
        axis_origin_m=(0.0, base.height_m, 0.0),
        axis_unit=(1.0, 0.0, 0.0),
        lower_coordinate_m=-100.0,
        upper_coordinate_m=100.0,
        regions=(
            GroundRegionalMaterialRegion(
                "far-region",
                1,
                50.0,
                60.0,
                region_surface,
            ),
        ),
        provenance=GroundProvenance(
            "pytest",
            "1.0",
            "pipeline-plan",
            "b" * 64,
        ),
    )


def _empty_termination_pair(
    reason: BounceTerminationReason,
) -> RepeatedBounceRequestResultPair:
    pair = execute_repeated_bounce_from_flight(
        _crossing_result(),
        _launch(),
        _settings(),
        is_cancelled=lambda: True,
    )
    if reason is BounceTerminationReason.CANCELLED:
        return pair
    request = pair.request
    result = RepeatedBounceResult(
        request_id=request.request_id,
        surface_id=request.surface_id,
        frame=request.frame,
        model_id=request.model_id,
        model_version=request.model_version,
        request_fingerprint_sha256=request.ground_request_sha256,
        trajectory=(),
        events=(),
        impacts=(),
        airborne_segments=(),
        handoff_state=None,
        termination=BounceTermination(
            reason,
            pair.result.termination.time_s,
            0.0,
        ),
        warnings=(),
    )
    return RepeatedBounceRequestResultPair(request, result)


def _time_limit_pair() -> RepeatedBounceRequestResultPair:
    pair = execute_repeated_bounce_from_flight(
        _crossing_result(),
        _launch(),
        _settings(max_time_s=0.001, output_interval_s=0.0001),
    )
    if pair.result.termination.reason is not BounceTerminationReason.TIME_LIMIT:
        raise AssertionError("time-limit fixture did not produce TIME_LIMIT")
    return pair
