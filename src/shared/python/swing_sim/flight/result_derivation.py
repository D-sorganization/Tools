"""Pure calculations for the canonical ball-flight result contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .result_contract import (
    AvailabilityReason,
    FlightMetricId,
    ValueStatus,
    flight_metric_catalog,
)
from .result_metrics import (
    FlightMetricInputs,
    FlightMetricResult,
    FlightMetricValue,
    FlightRunManifest,
    MetricNumber,
    MetricTrajectoryPoint,
    Vector3,
)
from .spin_axis_convention import spin_axis_tilt_deg

_MIN_SPEED_M_S = 1e-12


@dataclass(frozen=True)
class _LandingState:
    point: MetricTrajectoryPoint
    segment_end: int


def _landing_state(points: tuple[MetricTrajectoryPoint, ...]) -> _LandingState | None:
    was_airborne = False
    for index, (first, second) in enumerate(
        zip(points, points[1:], strict=False), start=1
    ):
        was_airborne = was_airborne or first.position_m[1] > 0.0
        if not was_airborne or second.position_m[1] > 0.0:
            continue
        denominator = first.position_m[1] - second.position_m[1]
        fraction = 1.0 if denominator == 0.0 else first.position_m[1] / denominator
        position = _lerp_vector(first.position_m, second.position_m, fraction)
        velocity = _lerp_vector(first.velocity_m_s, second.velocity_m_s, fraction)
        time_s = first.time_s + fraction * (second.time_s - first.time_s)
        return _LandingState(MetricTrajectoryPoint(time_s, position, velocity), index)
    return None


def _lerp_vector(first: Vector3, second: Vector3, fraction: float) -> Vector3:
    return tuple(a + fraction * (b - a) for a, b in zip(first, second, strict=True))  # type: ignore[return-value]


def _norm(value: Vector3) -> float:
    return math.sqrt(sum(component * component for component in value))


def _available(
    metric_id: FlightMetricId, numeric: MetricNumber, provenance: str
) -> FlightMetricValue:
    status = flight_metric_catalog().definition(metric_id).default_status
    return FlightMetricValue(metric_id, status, numeric, None, provenance)


def _unavailable(
    metric_id: FlightMetricId, reason: AvailabilityReason
) -> FlightMetricValue:
    provenance = flight_metric_catalog().definition(metric_id).provenance
    return FlightMetricValue(
        metric_id, ValueStatus.UNAVAILABLE, None, reason, provenance
    )


def _launch_values(
    inputs: FlightMetricInputs,
) -> dict[FlightMetricId, FlightMetricValue]:
    initial = inputs.trajectory[0].velocity_m_s
    horizontal = math.hypot(initial[0], initial[2])
    spin_magnitude = _norm(inputs.spin_vector_rpm)
    values = {
        FlightMetricId.INITIAL_VELOCITY: _available(
            FlightMetricId.INITIAL_VELOCITY, initial, "trajectory.initial_velocity"
        ),
        FlightMetricId.BALL_SPEED: _available(
            FlightMetricId.BALL_SPEED, _norm(initial), "derived.initial_velocity"
        ),
        FlightMetricId.VERTICAL_LAUNCH_ANGLE: _available(
            FlightMetricId.VERTICAL_LAUNCH_ANGLE,
            math.degrees(math.atan2(initial[1], horizontal)),
            "derived.initial_velocity",
        ),
        FlightMetricId.SPIN_VECTOR: _available(
            FlightMetricId.SPIN_VECTOR, inputs.spin_vector_rpm, "impact.spin_vector_rpm"
        ),
        FlightMetricId.TOTAL_SPIN: _available(
            FlightMetricId.TOTAL_SPIN, spin_magnitude, "derived.spin_vector_rpm"
        ),
    }
    if horizontal > _MIN_SPEED_M_S:
        values[FlightMetricId.LAUNCH_DIRECTION] = _available(
            FlightMetricId.LAUNCH_DIRECTION,
            math.degrees(math.atan2(initial[2], initial[0])),
            "derived.initial_velocity",
        )
    else:
        values[FlightMetricId.LAUNCH_DIRECTION] = _unavailable(
            FlightMetricId.LAUNCH_DIRECTION, AvailabilityReason.ZERO_HORIZONTAL_SPEED
        )
    spin_axis_tilt = spin_axis_tilt_deg(inputs.spin_vector_rpm)
    if spin_axis_tilt is not None:
        values[FlightMetricId.SPIN_AXIS_TILT] = _available(
            FlightMetricId.SPIN_AXIS_TILT,
            spin_axis_tilt,
            "derived.spin_vector_rpm",
        )
    else:
        values[FlightMetricId.SPIN_AXIS_TILT] = _unavailable(
            FlightMetricId.SPIN_AXIS_TILT, AvailabilityReason.ZERO_SPIN
        )
    return values


def _curve(points: tuple[MetricTrajectoryPoint, ...], heading: float) -> float:
    origin = points[0].position_m
    lateral = tuple(
        -math.sin(heading) * (point.position_m[0] - origin[0])
        + math.cos(heading) * (point.position_m[2] - origin[2])
        for point in points
    )
    return max(lateral, key=abs)


def _landing_values(
    inputs: FlightMetricInputs, landing: _LandingState
) -> dict[FlightMetricId, FlightMetricValue]:
    first = inputs.trajectory[0]
    point = landing.point
    delta_x = point.position_m[0] - first.position_m[0]
    delta_z = point.position_m[2] - first.position_m[2]
    horizontal = math.hypot(point.velocity_m_s[0], point.velocity_m_s[2])
    launch_horizontal = math.hypot(first.velocity_m_s[0], first.velocity_m_s[2])
    airborne = inputs.trajectory[: landing.segment_end] + (point,)
    values = {
        FlightMetricId.LANDING_POSITION: _available(
            FlightMetricId.LANDING_POSITION,
            point.position_m,
            "derived.linear_ground_interpolation",
        ),
        FlightMetricId.LANDING_VELOCITY: _available(
            FlightMetricId.LANDING_VELOCITY,
            point.velocity_m_s,
            "derived.linear_ground_interpolation",
        ),
        FlightMetricId.CARRY_DISTANCE: _available(
            FlightMetricId.CARRY_DISTANCE,
            math.hypot(delta_x, delta_z),
            "derived.landing_position",
        ),
        FlightMetricId.CARRY_OFFLINE: _available(
            FlightMetricId.CARRY_OFFLINE, delta_z, "derived.landing_position"
        ),
        FlightMetricId.APEX_HEIGHT: _available(
            FlightMetricId.APEX_HEIGHT,
            max(sample.position_m[1] for sample in airborne),
            "derived.trajectory_samples",
        ),
        FlightMetricId.FLIGHT_TIME: _available(
            FlightMetricId.FLIGHT_TIME,
            point.time_s - first.time_s,
            "derived.landing_time",
        ),
        FlightMetricId.TERMINAL_SPEED: _available(
            FlightMetricId.TERMINAL_SPEED,
            _norm(point.velocity_m_s),
            "derived.landing_velocity",
        ),
    }
    if horizontal > _MIN_SPEED_M_S:
        values[FlightMetricId.LANDING_ANGLE] = _available(
            FlightMetricId.LANDING_ANGLE,
            math.degrees(math.atan2(-point.velocity_m_s[1], horizontal)),
            "derived.landing_velocity",
        )
        values[FlightMetricId.TERMINAL_DIRECTION] = _available(
            FlightMetricId.TERMINAL_DIRECTION,
            math.degrees(math.atan2(point.velocity_m_s[2], point.velocity_m_s[0])),
            "derived.landing_velocity",
        )
    else:
        for metric_id in (
            FlightMetricId.LANDING_ANGLE,
            FlightMetricId.TERMINAL_DIRECTION,
        ):
            values[metric_id] = _unavailable(
                metric_id, AvailabilityReason.ZERO_HORIZONTAL_SPEED
            )
    if launch_horizontal > _MIN_SPEED_M_S:
        heading = math.atan2(first.velocity_m_s[2], first.velocity_m_s[0])
        values[FlightMetricId.CURVE] = _available(
            FlightMetricId.CURVE,
            _curve(airborne, heading),
            "derived.initial_vertical_plane",
        )
    else:
        values[FlightMetricId.CURVE] = _unavailable(
            FlightMetricId.CURVE, AvailabilityReason.ZERO_HORIZONTAL_SPEED
        )
    return values


def _target_values(
    inputs: FlightMetricInputs, landing: _LandingState
) -> dict[FlightMetricId, FlightMetricValue]:
    target = inputs.target_position_m
    metric_ids = (
        FlightMetricId.TARGET_RESIDUAL,
        FlightMetricId.TARGET_DOWNRANGE_RESIDUAL,
        FlightMetricId.TARGET_LATERAL_RESIDUAL,
    )
    if target is None:
        return {
            metric_id: _unavailable(metric_id, AvailabilityReason.TARGET_NOT_CONFIGURED)
            for metric_id in metric_ids
        }
    residual: Vector3 = (
        landing.point.position_m[0] - target[0],
        landing.point.position_m[1] - target[1],
        landing.point.position_m[2] - target[2],
    )
    return {
        FlightMetricId.TARGET_RESIDUAL: _available(
            FlightMetricId.TARGET_RESIDUAL, _norm(residual), "derived.target_residual"
        ),
        FlightMetricId.TARGET_DOWNRANGE_RESIDUAL: _available(
            FlightMetricId.TARGET_DOWNRANGE_RESIDUAL,
            residual[0],
            "derived.target_residual",
        ),
        FlightMetricId.TARGET_LATERAL_RESIDUAL: _available(
            FlightMetricId.TARGET_LATERAL_RESIDUAL,
            residual[2],
            "derived.target_residual",
        ),
    }


def _ground_values(
    inputs: FlightMetricInputs,
) -> dict[FlightMetricId, FlightMetricValue]:
    ids = (
        FlightMetricId.TOTAL_DISTANCE,
        FlightMetricId.ROLL_DISTANCE,
        FlightMetricId.BOUNCE_COUNT,
        FlightMetricId.FINAL_OFFLINE,
    )
    ground = inputs.ground_result
    if ground is None:
        return {
            metric_id: _unavailable(metric_id, AvailabilityReason.GROUND_MODEL_REQUIRED)
            for metric_id in ids
        }
    numerics = (
        ground.total_distance_m,
        ground.roll_distance_m,
        ground.bounce_count,
        ground.final_offline_m,
    )
    return {
        metric_id: _available(metric_id, numeric, ground.model_id)
        for metric_id, numeric in zip(ids, numerics, strict=True)
    }


def derive_flight_metric_result(
    inputs: FlightMetricInputs, manifest: FlightRunManifest
) -> FlightMetricResult:
    """Derive every canonical value without fabricating missing model output."""
    if not inputs.trajectory:
        empty_values = tuple(
            _unavailable(metric_id, AvailabilityReason.INSUFFICIENT_TRAJECTORY)
            for metric_id in FlightMetricId
        )
        return FlightMetricResult(manifest, empty_values)
    values = _launch_values(inputs)
    landing = _landing_state(inputs.trajectory)
    landing_ids: set[FlightMetricId] = set(FlightMetricId)
    landing_ids.difference_update(
        values,
        {
            FlightMetricId.TOTAL_DISTANCE,
            FlightMetricId.ROLL_DISTANCE,
            FlightMetricId.BOUNCE_COUNT,
            FlightMetricId.FINAL_OFFLINE,
        },
    )
    if landing is None:
        reason = (
            AvailabilityReason.INSUFFICIENT_TRAJECTORY
            if len(inputs.trajectory) < 2
            else AvailabilityReason.NO_GROUND_CROSSING
        )
        values.update(
            {metric_id: _unavailable(metric_id, reason) for metric_id in landing_ids}
        )
        if inputs.target_position_m is None:
            target_ids = (
                FlightMetricId.TARGET_RESIDUAL,
                FlightMetricId.TARGET_DOWNRANGE_RESIDUAL,
                FlightMetricId.TARGET_LATERAL_RESIDUAL,
            )
            values.update(
                {
                    metric_id: _unavailable(
                        metric_id, AvailabilityReason.TARGET_NOT_CONFIGURED
                    )
                    for metric_id in target_ids
                }
            )
    else:
        values.update(_landing_values(inputs, landing))
        values.update(_target_values(inputs, landing))
    values.update(_ground_values(inputs))
    return FlightMetricResult(manifest, tuple(values.values()))


__all__ = ["derive_flight_metric_result"]
