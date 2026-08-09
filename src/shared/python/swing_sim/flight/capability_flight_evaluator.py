"""Model-backed capability evaluator for robust launch optimization."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .capability_contract import (
    ClubCapability,
    OptimizationRequest,
    PlayerCapabilityProfile,
)
from .frames import from_flight_frame
from .inverse_contract import (
    EvaluatedMetric,
    EvaluationStatus,
    SolverEvaluation,
)
from .pipeline import simulate
from .result_contract import FlightMetricId, ValueStatus
from .result_metrics import (
    FlightMetricInputs,
    FlightRunManifest,
    MetricTrajectoryPoint,
    derive_flight_metric_result,
)
from .types import LaunchConditions

_MODEL_ID = "waterloo_penner"
_MODEL_VERSION = "waterloo-penner-coefficients/v1"
_PROVENANCE = f"{_MODEL_ID}:{_MODEL_VERSION}:scipy-rk45"
_REQUIRED_PARAMETERS = frozenset({"ball_speed", "launch_angle", "launch_direction"})
_OPTIONAL_PARAMETERS = frozenset({"total_spin", "spin_axis_tilt"})
_EXPECTED_UNITS = {
    "ball_speed": "m/s",
    "launch_angle": "deg",
    "launch_direction": "deg",
    "total_spin": "rpm",
    "spin_axis_tilt": "deg",
}
_PHYSICAL_DOMAINS = {
    "ball_speed": (0.0, math.inf, False),
    "launch_angle": (-90.0, 90.0, True),
    "launch_direction": (-180.0, 180.0, True),
    "total_spin": (0.0, math.inf, True),
    "spin_axis_tilt": (-90.0, 90.0, True),
}
_RAD_S_TO_RPM = 60.0 / (2.0 * math.pi)
_GROUND_HEIGHT_TOLERANCE_M = 1e-9


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


@dataclass(frozen=True)
class CapabilitySpinDefault:
    """Auditable fixed spin used when one club profile omits spin variables."""

    club_id: str
    total_spin_rpm: float
    spin_axis_tilt_deg: float
    provenance: str

    def __post_init__(self) -> None:
        if not self.club_id.strip() or not self.provenance.strip():
            raise ValueError("spin default club_id and provenance must be nonempty")
        spin = _finite(self.total_spin_rpm, "total_spin_rpm")
        tilt = _finite(self.spin_axis_tilt_deg, "spin_axis_tilt_deg")
        _validate_physical_value("total_spin", spin)
        _validate_physical_value("spin_axis_tilt", tilt)
        object.__setattr__(self, "club_id", self.club_id.strip())
        object.__setattr__(self, "total_spin_rpm", spin)
        object.__setattr__(self, "spin_axis_tilt_deg", tilt)
        object.__setattr__(self, "provenance", self.provenance.strip())


@dataclass(frozen=True)
class CapabilityFlightEvaluatorConfig:
    """Cross-runtime integration controls and explicit per-club spin defaults."""

    max_time_s: float = 10.0
    trajectory_sample_interval_s: float = 0.01
    spin_defaults: tuple[CapabilitySpinDefault, ...] = ()

    def __post_init__(self) -> None:
        max_time = _finite(self.max_time_s, "max_time_s")
        interval = _finite(
            self.trajectory_sample_interval_s, "trajectory_sample_interval_s"
        )
        if max_time <= 0.0:
            raise ValueError("max_time_s must be > 0")
        if not 0.001 <= interval <= 0.1:
            raise ValueError(
                "trajectory_sample_interval_s must lie within [0.001, 0.1]"
            )
        ratio = interval / 0.001
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                "trajectory_sample_interval_s must align to the 0.001 s step"
            )
        defaults = tuple(self.spin_defaults)
        identifiers = tuple(item.club_id for item in defaults)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("spin default club_ids must be unique")
        object.__setattr__(self, "max_time_s", max_time)
        object.__setattr__(self, "trajectory_sample_interval_s", interval)
        object.__setattr__(self, "spin_defaults", defaults)


@dataclass(frozen=True)
class _EvaluatorBinding:
    clubs: dict[str, ClubCapability]
    spin_defaults: dict[str, CapabilitySpinDefault]
    target_position_m: tuple[float, float, float]
    config: CapabilityFlightEvaluatorConfig


def _validate_club(
    club: ClubCapability, spin_defaults: Mapping[str, CapabilitySpinDefault]
) -> None:
    units = {item.parameter_id: item.unit for item in club.parameters}
    fields = set(units)
    if not _REQUIRED_PARAMETERS <= fields:
        missing = sorted(_REQUIRED_PARAMETERS - fields)
        raise ValueError(f"{club.club_id} is missing required parameters: {missing}")
    if not fields <= _REQUIRED_PARAMETERS | _OPTIONAL_PARAMETERS:
        raise ValueError(f"{club.club_id} declares unsupported capability parameters")
    declared_spin = fields & _OPTIONAL_PARAMETERS
    if declared_spin and declared_spin != _OPTIONAL_PARAMETERS:
        raise ValueError("total_spin and spin_axis_tilt must be declared together")
    if not declared_spin and club.club_id not in spin_defaults:
        raise ValueError(f"{club.club_id} requires an explicit spin default")
    for parameter_id, unit in units.items():
        expected = _EXPECTED_UNITS[parameter_id]
        if unit != expected:
            raise ValueError(f"{parameter_id} must use {expected}, not {unit}")
        parameter = next(
            item for item in club.parameters if item.parameter_id == parameter_id
        )
        _validate_physical_value(parameter_id, parameter.lower_bound)
        _validate_physical_value(parameter_id, parameter.upper_bound)


def _validate_physical_value(parameter_id: str, value: float) -> None:
    lower, upper, inclusive_lower = _PHYSICAL_DOMAINS[parameter_id]
    lower_valid = value >= lower if inclusive_lower else value > lower
    if lower_valid and value <= upper:
        return
    if parameter_id == "ball_speed":
        raise ValueError("ball_speed must be greater than zero")
    raise ValueError(f"{parameter_id} must lie within [{lower:g}, {upper:g}]")


def _binding(
    profile: PlayerCapabilityProfile,
    request: OptimizationRequest,
    config: CapabilityFlightEvaluatorConfig,
) -> _EvaluatorBinding:
    spin_defaults = {item.club_id: item for item in config.spin_defaults}
    clubs = {club_id: profile.club(club_id) for club_id in request.club_ids}
    for club in clubs.values():
        _validate_club(club, spin_defaults)
    target = (request.target.distance_m, 0.0, request.target.lateral_m)
    return _EvaluatorBinding(clubs, spin_defaults, target, config)


def _validated_sample(
    club: ClubCapability, values: Mapping[str, float]
) -> dict[str, float]:
    parameters = {item.parameter_id: item for item in club.parameters}
    if set(values) != set(parameters):
        raise ValueError("capability sample fields do not match the club profile")
    parsed = {key: _finite(value, key) for key, value in values.items()}
    for parameter_id, value in parsed.items():
        parameter = parameters[parameter_id]
        _validate_physical_value(parameter_id, value)
        if not parameter.lower_bound <= value <= parameter.upper_bound:
            raise ValueError(f"{parameter_id} lies outside declared safe bounds")
    return parsed


def _launch(
    values: Mapping[str, float], spin_default: CapabilitySpinDefault | None
) -> tuple[LaunchConditions, str]:
    spin = values.get(
        "total_spin", spin_default.total_spin_rpm if spin_default else math.nan
    )
    tilt = values.get(
        "spin_axis_tilt", spin_default.spin_axis_tilt_deg if spin_default else math.nan
    )
    spin_source = (
        "sampled_profile"
        if spin_default is None
        else f"fixed_club_default:{spin_default.provenance}"
    )
    tilt_radians = math.radians(tilt)
    return (
        LaunchConditions(
            ball_speed=values["ball_speed"],
            launch_angle=math.radians(values["launch_angle"]),
            azimuth_angle=-math.radians(values["launch_direction"]),
            spin_rate=spin,
            # Positive target-frame tilt is toward fade/right: (0, -sin, cos).
            spin_axis=(0.0, -math.cos(tilt_radians), -math.sin(tilt_radians)),
        ),
        spin_source,
    )


def _metric_inputs(
    launch: LaunchConditions,
    binding: _EvaluatorBinding,
) -> FlightMetricInputs:
    config = binding.config
    flight = simulate(
        launch,
        model_name=_MODEL_ID,
        max_time=config.max_time_s,
        dt=config.trajectory_sample_interval_s,
    )
    trajectory_points: list[MetricTrajectoryPoint] = []
    for point in flight.trajectory:
        position = from_flight_frame(point.position)
        if abs(position[1]) <= _GROUND_HEIGHT_TOLERANCE_M:
            position[1] = 0.0
        trajectory_points.append(
            MetricTrajectoryPoint(
                point.time,
                tuple(position),
                tuple(from_flight_frame(point.velocity)),
            )
        )
    trajectory = tuple(trajectory_points)
    spin_rpm = tuple(from_flight_frame(launch.get_spin_vector()) * _RAD_S_TO_RPM)
    return FlightMetricInputs(trajectory, spin_rpm, binding.target_position_m)


def _has_ground_crossing(inputs: FlightMetricInputs) -> bool:
    heights = tuple(point.position_m[1] for point in inputs.trajectory)
    return any(
        first > 0.0 and second <= 0.0
        for first, second in zip(heights, heights[1:], strict=False)
    )


def _manifest(crossed: bool) -> FlightRunManifest:
    return FlightRunManifest(
        model_id=_MODEL_ID,
        model_version=_MODEL_VERSION,
        integration_status="complete" if crossed else "nonconverged",
        termination_reason="ground_crossing" if crossed else "max_time_reached",
        environment=(("air_model", "standard"), ("integrator", "scipy-rk45")),
        wind=(("model", "still_air"),),
        uncertainty_status="deterministic",
    )


def _as_evaluation(inputs: FlightMetricInputs, spin_source: str) -> SolverEvaluation:
    crossed = _has_ground_crossing(inputs)
    result = derive_flight_metric_result(inputs, _manifest(crossed))
    required = (FlightMetricId.CARRY_DISTANCE, FlightMetricId.CARRY_OFFLINE)
    if not crossed or any(
        result.value(metric_id).status is ValueStatus.UNAVAILABLE
        for metric_id in required
    ):
        return SolverEvaluation(
            EvaluationStatus.NONCONVERGED,
            (),
            "no_ground_crossing_before_max_time",
        )
    metrics = tuple(
        EvaluatedMetric(
            value.metric_id,
            value.numeric,
            f"ball-flight-result/v1|{_PROVENANCE}|spin:{spin_source}|{value.provenance}",
        )
        for value in result.values
        if isinstance(value.numeric, float)
    )
    return SolverEvaluation(EvaluationStatus.COMPLETE, metrics)


class _ModelBackedCapabilityEvaluator:
    def __init__(self, binding: _EvaluatorBinding) -> None:
        self._binding = binding

    def __call__(
        self, club_id: str, parameters: Mapping[str, float]
    ) -> SolverEvaluation:
        """Evaluate one validated club sample through the full flight model."""
        if club_id not in self._binding.clubs:
            raise ValueError(f"unknown requested club_id: {club_id}")
        club = self._binding.clubs[club_id]
        values = _validated_sample(club, parameters)
        launch, spin_source = _launch(values, self._binding.spin_defaults.get(club_id))
        try:
            return _as_evaluation(_metric_inputs(launch, self._binding), spin_source)
        except (FloatingPointError, OverflowError):
            return SolverEvaluation(EvaluationStatus.FAILED, (), "flight_model_failure")


def make_capability_flight_evaluator(
    profile: PlayerCapabilityProfile,
    request: OptimizationRequest,
    config: CapabilityFlightEvaluatorConfig | None = None,
) -> Callable[[str, Mapping[str, float]], SolverEvaluation]:
    """Bind a profile/request to an optimizer-compatible full-flight evaluator.

    Three-variable capability profiles require an explicit, sourced per-club
    spin default. Clubs may instead declare and vary both total_spin and
    spin_axis_tilt.
    """
    active_config = config or CapabilityFlightEvaluatorConfig()
    return _ModelBackedCapabilityEvaluator(_binding(profile, request, active_config))


__all__ = [
    "CapabilityFlightEvaluatorConfig",
    "CapabilitySpinDefault",
    "make_capability_flight_evaluator",
]
