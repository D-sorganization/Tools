"""Atomic, immutable accepted-flight evidence for the PyQt explorer."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

import numpy as np

from rate_of_closure.flight_sample_inspector import (
    MAX_FLIGHT_POSITION_M,
    MAX_FLIGHT_TIME_S,
    FlightSamplePlan,
    FlightSampleSeries,
    plan_flight_samples,
)
from rate_of_closure.model import MPH_PER_MPS
from rate_of_closure.simulation.flight_explorer import (
    EXPLORER_METRIC_KEYS,
    FlightExploration,
    WindComparison,
    launch_from_delivery,
    launch_from_direct,
)
from shared.python.swing_sim.flight import (
    LaunchConditions,
    LaunchDirectionConvention,
    WindScenario,
)
from shared.python.swing_sim.flight.registry import FlightModelType
from shared.python.swing_sim.impact import DeliveryParameters

DIRECT_INPUT_KEYS = (
    "ball_speed_mph",
    "launch_angle_deg",
    "launch_direction_deg",
    "spin_rpm",
    "spin_axis_tilt_deg",
)
DELIVERY_INPUT_KEYS = (
    "clubhead_speed_mps",
    "club_path_deg",
    "face_angle_deg",
    "attack_angle_deg",
    "dynamic_loft_deg",
    "impact_offset_toe_mm",
    "impact_offset_high_mm",
    "lie_deg",
)


@dataclass(frozen=True)
class FlightStudyContext:
    """Complete normalized producing request, including resolved launch."""

    entry_mode: str
    input_values: tuple[tuple[str, float], ...]
    direction_convention: LaunchDirectionConvention
    model_name: str
    wind_scenario: WindScenario | None
    expected_launch: LaunchConditions
    kernel_revision: str = "python-rk45-v1"

    def __post_init__(self) -> None:
        expected_keys = (
            DIRECT_INPUT_KEYS if self.entry_mode == "direct" else DELIVERY_INPUT_KEYS
        )
        if self.entry_mode not in {"direct", "delivery"}:
            raise ValueError("accepted flight entry mode is unknown")
        if tuple(key for key, _value in self.input_values) != expected_keys:
            raise ValueError("accepted flight input identity is incomplete")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for _key, value in self.input_values
        ):
            raise ValueError("accepted flight inputs must be finite real numbers")
        if not isinstance(self.direction_convention, LaunchDirectionConvention):
            raise ValueError("accepted launch direction convention is unknown")
        if self.model_name not in {model.value for model in FlightModelType}:
            raise ValueError("accepted flight model is unknown")
        if self.kernel_revision != "python-rk45-v1":
            raise ValueError("accepted flight kernel revision is unknown")
        if not isinstance(self.expected_launch, LaunchConditions):
            raise ValueError("accepted flight must bind resolved launch authority")
        if self.expected_launch.wind_scenario != self.wind_scenario:
            raise ValueError("resolved launch wind disagrees with accepted context")
        if self.expected_launch.wind_speed != 0.0:
            raise ValueError("legacy scalar wind is outside the explorer authority")
        values = dict(self.input_values)
        if self.entry_mode == "direct":
            rebuilt = launch_from_direct(
                values["ball_speed_mph"],
                values["launch_angle_deg"],
                values["launch_direction_deg"],
                values["spin_rpm"],
                values["spin_axis_tilt_deg"],
                direction_convention=self.direction_convention,
            )
        else:
            rebuilt = launch_from_delivery(
                DeliveryParameters(
                    clubhead_speed_mps=values["clubhead_speed_mps"],
                    club_path_deg=values["club_path_deg"],
                    face_angle_deg=values["face_angle_deg"],
                    attack_angle_deg=values["attack_angle_deg"],
                    dynamic_loft_deg=values["dynamic_loft_deg"],
                    impact_offset_toe_mm=values["impact_offset_toe_mm"],
                    impact_offset_high_mm=values["impact_offset_high_mm"],
                    lie_deg=values["lie_deg"],
                )
            )
        rebuilt = replace(rebuilt, wind_speed=0.0, wind_scenario=self.wind_scenario)
        if rebuilt != self.expected_launch:
            raise ValueError(
                "raw flight inputs disagree with resolved launch authority"
            )

    def label(self) -> str:
        inputs = ", ".join(f"{key} {value:.3f}" for key, value in self.input_values)
        wind = (
            "calm"
            if self.wind_scenario is None
            else "wind scenario "
            f"{self.wind_scenario.schema_version}; "
            f"{len(self.wind_scenario.gusts)} gusts; "
            f"{str(self.wind_scenario.provenance)[:96]}"
        )
        convention = (
            f"; {self.direction_convention.value}"
            if self.entry_mode == "direct"
            else ""
        )
        return (
            f"{self.entry_mode}; {inputs}{convention}; {wind}; "
            f"model {self.model_name}; {self.kernel_revision}"
        )


@dataclass(frozen=True)
class AcceptedFlightStudy:
    """Complete candidate published only after every derivation succeeds."""

    generation: int
    context: FlightStudyContext
    exploration: FlightExploration
    calm_comparison: FlightExploration | None
    comparison: WindComparison | None
    plan: FlightSamplePlan


def _immutable_array(value: object, shape: tuple[int, ...], field: str) -> np.ndarray:
    raw = np.asarray(value)
    if (
        raw.shape != shape
        or raw.dtype.kind not in "fiu"
        or not np.all(np.isfinite(raw))
    ):
        raise ValueError(f"{field} must contain finite aligned numeric evidence")
    if np.any(
        np.abs(raw) > (1_000 if field == "velocities" else MAX_FLIGHT_POSITION_M)
    ):
        raise ValueError(f"{field} exceeds the explorer evidence envelope")
    source = np.asarray(raw, dtype=np.float64)
    return cast(
        np.ndarray,
        np.frombuffer(source.tobytes(), dtype=np.float64).reshape(source.shape),
    )


def _snapshot(exploration: FlightExploration) -> FlightExploration:
    if not isinstance(exploration, FlightExploration):
        raise ValueError("accepted flight result must be a FlightExploration")
    series = FlightSampleSeries.from_exploration(exploration)
    count = len(series.times_s)
    velocities = _immutable_array(exploration.velocities, (count, 3), "velocities")
    metrics = exploration.metrics
    if not isinstance(metrics, dict) or set(metrics) != set(EXPLORER_METRIC_KEYS):
        raise ValueError("flight summary must contain the exact explorer metrics")
    copied: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("flight summary metrics must be finite numbers")
        copied[key] = float(value)
    if any(not math.isfinite(value) for value in copied.values()):
        raise ValueError("flight summary metrics must be finite numbers")
    times = np.frombuffer(np.asarray(series.times_s).tobytes(), dtype=np.float64)
    positions = _immutable_array(series.positions_m, (count, 3), "positions")
    return FlightExploration(
        exploration.launch,
        str(exploration.model_name),
        times,
        positions,
        velocities,
        cast(dict[str, float], MappingProxyType(copied)),
    )


def _close(actual: float, expected: float, field: str) -> None:
    tolerance = 1e-7 * max(1.0, abs(actual), abs(expected))
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{field} disagrees with exact trajectory evidence")


def _validate(exploration: FlightExploration, plan: FlightSamplePlan) -> None:
    first, last = plan.raw_sample(0), plan.raw_sample(plan.raw_count - 1)
    metrics: Mapping[str, float] = exploration.metrics
    _close(first.time_s, 0.0, "first sample time")
    _close(first.downrange_m, 0.0, "launch downrange position")
    _close(first.height_m, exploration.launch.ball_radius, "launch height")
    _close(first.right_m, 0.0, "launch lateral position")
    _close(last.height_m, exploration.launch.ball_radius, "landing height")
    if any(
        sample.height_m < exploration.launch.ball_radius - 1e-7
        for sample in plan.samples
    ):
        raise ValueError("flight evidence falls below the canonical ground plane")
    _close(metrics["flight_time_s"], last.time_s, "flight time")
    _close(
        metrics["carry_m"],
        math.hypot(last.downrange_m - first.downrange_m, last.right_m - first.right_m),
        "carry",
    )
    _close(metrics["lateral_m"], last.right_m - first.right_m, "lateral offset")
    sampled_height = max(sample.height_m - first.height_m for sample in plan.samples)
    _close(metrics["max_height_m"], sampled_height, "maximum height")
    if (
        not 0
        <= metrics["carry_m"]
        <= math.hypot(MAX_FLIGHT_POSITION_M, MAX_FLIGHT_POSITION_M)
    ):
        raise ValueError("carry is outside the explorer evidence envelope")
    if not 0 <= metrics["flight_time_s"] <= MAX_FLIGHT_TIME_S:
        raise ValueError("flight time is outside the explorer evidence envelope")
    _close(
        metrics["launch_direction_deg"],
        metrics["launch_azimuth_deg"],
        "direction alias",
    )
    _close(
        metrics["ball_speed_mph"],
        exploration.launch.ball_speed * MPH_PER_MPS,
        "ball speed",
    )
    _close(
        metrics["launch_angle_deg"],
        math.degrees(exploration.launch.launch_angle),
        "launch angle",
    )
    _close(
        metrics["launch_direction_deg"],
        -math.degrees(exploration.launch.azimuth_angle),
        "launch direction",
    )
    _close(metrics["spin_rpm"], exploration.launch.spin_rate, "spin rate")
    launch_velocity = exploration.velocities[0]
    launch_horizontal = math.hypot(float(launch_velocity[0]), float(launch_velocity[2]))
    _close(
        float(np.linalg.norm(launch_velocity)),
        exploration.launch.ball_speed,
        "raw launch speed",
    )
    _close(
        math.atan2(float(launch_velocity[1]), launch_horizontal),
        exploration.launch.launch_angle,
        "raw launch angle",
    )
    _close(
        math.atan2(float(launch_velocity[2]), float(launch_velocity[0])),
        -exploration.launch.azimuth_angle,
        "raw launch direction",
    )
    landing_velocity = exploration.velocities[-1]
    horizontal = math.hypot(float(landing_velocity[0]), float(landing_velocity[2]))
    landing_angle = (
        math.degrees(math.atan2(-float(landing_velocity[1]), horizontal))
        if horizontal > 0.1
        else 90.0
    )
    _close(metrics["landing_angle_deg"], landing_angle, "landing angle")


def _same_exploration(left: FlightExploration, right: FlightExploration) -> bool:
    return bool(
        left.launch == right.launch
        and left.model_name == right.model_name
        and left.metrics == right.metrics
        and np.array_equal(left.times, right.times)
        and np.array_equal(left.positions, right.positions)
        and np.array_equal(left.velocities, right.velocities)
    )


def build_accepted_flight_study(
    generation: int,
    context: FlightStudyContext,
    exploration_input: FlightExploration,
    comparison_input: WindComparison | None,
) -> AcceptedFlightStudy:
    """Validate and snapshot a complete candidate before publication."""
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or not 1 <= generation <= 2**53 - 1
    ):
        raise ValueError("accepted flight generation must be a positive safe integer")
    exploration = _snapshot(exploration_input)
    if (
        exploration.launch != context.expected_launch
        or exploration.model_name != context.model_name
    ):
        raise ValueError("flight result authority disagrees with accepted request")
    plan = plan_flight_samples(FlightSampleSeries.from_exploration(exploration))
    _validate(exploration, plan)
    calm: FlightExploration | None = None
    comparison: WindComparison | None = None
    if comparison_input is not None:
        if (
            context.wind_scenario is None
            or comparison_input.scenario != context.wind_scenario
        ):
            raise ValueError("wind comparison scenario disagrees with accepted request")
        wind = _snapshot(comparison_input.wind)
        calm = _snapshot(comparison_input.calm)
        if not _same_exploration(wind, exploration):
            raise ValueError(
                "wind comparison primary disagrees with accepted exploration"
            )
        expected_calm = replace(
            context.expected_launch, wind_speed=0.0, wind_scenario=None
        )
        if calm.launch != expected_calm or calm.model_name != context.model_name:
            raise ValueError(
                "calm comparison does not share the accepted launch authority"
            )
        calm_plan = plan_flight_samples(FlightSampleSeries.from_exploration(calm))
        _validate(calm, calm_plan)
        delta_keys = {
            "carry_m",
            "max_height_m",
            "flight_time_s",
            "landing_angle_deg",
            "lateral_m",
        }
        if set(comparison_input.deltas) != delta_keys:
            raise ValueError("wind comparison deltas are incomplete")
        deltas: dict[str, float] = {}
        for key in delta_keys:
            value = comparison_input.deltas[key]
            if not math.isfinite(value):
                raise ValueError("wind comparison deltas must be finite")
            _close(value, wind.metrics[key] - calm.metrics[key], f"wind {key}")
            deltas[key] = float(value)
        comparison = WindComparison(
            calm,
            wind,
            context.wind_scenario,
            cast(dict[str, float], MappingProxyType(deltas)),
        )
    elif context.wind_scenario is not None:
        raise ValueError("enabled wind request requires a cohesive comparison")
    return AcceptedFlightStudy(generation, context, exploration, calm, comparison, plan)
