"""Validated leaf records for the regional-ground execution-job contract."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from rate_of_closure.application._workspace_validation import exact_mapping, stable_id
from shared.python.swing_sim.ball_setup import (
    HEIGHT_REFERENCE,
    BallSetup,
    BallSupportMode,
)
from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_json
from shared.python.swing_sim.flight import (
    FlightResult,
    FlightStatePoint,
    LaunchConditions,
)

FLIGHT_FRAME = "flight_frame:x_forward,y_left,z_up"
MAX_CAPTURE_SPEED_M_S = 100.0
MAX_EXECUTION_PARALLELISM = 32
MAX_EXECUTION_TIMEOUT_S = 3_600.0
MAX_FLIGHT_SETTINGS = 64

_BALL_SETUP_FIELDS = frozenset(
    {"support_mode", "tee_height_m", "height_reference", "ball_center_m"}
)
_EXECUTION_FIELDS = frozenset(
    {"max_trials", "max_parallelism", "timeout_s", "fail_fast"}
)
_FLIGHT_FIELDS = frozenset(
    {"model_id", "model_version", "settings", "trajectory_sha256", "result_sha256"}
)
_LAUNCH_FIELDS = frozenset(
    {
        "frame",
        "ball_speed_m_s",
        "launch_angle_rad",
        "azimuth_angle_rad",
        "spin_rate_rpm",
        "spin_axis_unit",
        "ball_mass_kg",
        "ball_radius_m",
        "air_density_kg_m3",
        "gravity_m_s2",
        "wind_speed_m_s",
        "wind_direction_rad",
        "ball_setup",
    }
)


def digest(value: object, name: str) -> str:
    """Return one canonical lowercase SHA-256 digest."""
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def sha256(payload: object) -> str:
    """Return the shared canonical-numeric JSON digest for a payload."""
    text = str(canonical_numeric_json(payload))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_text(value: object, name: str) -> str:
    """Return bounded nonblank canonical text."""
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty canonical text")
    if len(value) > 128:
        raise ValueError(f"{name} exceeds 128 characters")
    canonical_numeric_json(value)
    return value


def finite(value: object, name: str) -> float:
    """Return a finite cross-runtime-safe non-Boolean number."""
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise ValueError(f"{name} must be finite")
    canonical_numeric_json(value)
    number = cast(int | float, value)
    return float(number)


def positive(value: object, name: str, maximum: float) -> float:
    """Return a finite number inside one explicit positive bound."""
    number = finite(value, name)
    if not 0.0 < number <= maximum:
        raise ValueError(f"{name} must lie within (0, {maximum:g}]")
    return number


def integer(value: object, name: str, minimum: int, maximum: int) -> int:
    """Return a non-Boolean integer inside an explicit inclusive bound."""
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must lie within [{minimum}, {maximum}]")
    return value


def vector(value: object, name: str) -> tuple[float, float, float]:
    """Return a strict finite three-component vector."""
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must contain three components")
    try:
        items = tuple(cast(Any, value))
    except TypeError as exc:
        raise ValueError(f"{name} must contain three components") from exc
    if len(items) != 3:
        raise ValueError(f"{name} must contain three components")
    return cast(tuple[float, float, float], tuple(finite(item, name) for item in items))


def _ball_setup_from_dict(value: object) -> BallSetup:
    data = exact_mapping(value, _BALL_SETUP_FIELDS, "ball_setup")
    if data["height_reference"] != HEIGHT_REFERENCE:
        raise ValueError("unsupported ball_setup height_reference")
    setup = BallSetup(
        BallSupportMode(canonical_text(data["support_mode"], "support_mode")),
        finite(data["tee_height_m"], "tee_height_m"),
    )
    if vector(data["ball_center_m"], "ball_center_m") != setup.ball_center_m:
        raise ValueError("ball_center_m must match the derived ball setup geometry")
    return setup


def _spin_axis(launch: LaunchConditions) -> tuple[float, float, float]:
    if launch.spin_axis is not None:
        axis = cast(tuple[float, float, float], launch.spin_axis)
        return axis
    if launch.spin_rate == 0.0:
        return (0.0, -1.0, 0.0)
    raw = launch.get_spin_vector()
    magnitude = math.sqrt(sum(float(item) ** 2 for item in raw))
    return cast(
        tuple[float, float, float], tuple(float(item) / magnitude for item in raw)
    )


@dataclass(frozen=True)
class FlightLaunchInput:
    """Exact constant-wind launch state consumed by current flight models."""

    launch: LaunchConditions

    def __post_init__(self) -> None:
        if type(self.launch) is not LaunchConditions:
            raise TypeError("launch must be an exact LaunchConditions")
        if self.launch.wind_scenario is not None:
            raise ValueError("execution-job/v1 requires resolved constant wind")
        if self.launch.wind_speed < 0.0:
            raise ValueError("wind_speed_m_s must be nonnegative")

    @property
    def ball_setup(self) -> BallSetup:
        """Return the launch-owned ball setup."""
        return self.launch.ball_setup

    def to_dict(self) -> dict[str, Any]:
        """Return the exact SI launch mapping."""
        launch = self.launch
        return {
            "frame": FLIGHT_FRAME,
            "ball_speed_m_s": launch.ball_speed,
            "launch_angle_rad": launch.launch_angle,
            "azimuth_angle_rad": launch.azimuth_angle,
            "spin_rate_rpm": launch.spin_rate,
            "spin_axis_unit": list(_spin_axis(launch)),
            "ball_mass_kg": launch.ball_mass,
            "ball_radius_m": launch.ball_radius,
            "air_density_kg_m3": launch.air_density,
            "gravity_m_s2": launch.gravity,
            "wind_speed_m_s": launch.wind_speed,
            "wind_direction_rad": launch.wind_direction,
            "ball_setup": launch.ball_setup.to_json_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> FlightLaunchInput:
        """Parse an exact launch mapping without running flight physics."""
        data = exact_mapping(value, _LAUNCH_FIELDS, "launch")
        if data["frame"] != FLIGHT_FRAME:
            raise ValueError("unsupported launch frame")
        wind_speed = finite(data["wind_speed_m_s"], "wind_speed_m_s")
        if wind_speed < 0.0:
            raise ValueError("wind_speed_m_s must be nonnegative")
        return cls(
            LaunchConditions(
                ball_speed=finite(data["ball_speed_m_s"], "ball_speed_m_s"),
                launch_angle=finite(data["launch_angle_rad"], "launch_angle_rad"),
                azimuth_angle=finite(data["azimuth_angle_rad"], "azimuth_angle_rad"),
                spin_rate=finite(data["spin_rate_rpm"], "spin_rate_rpm"),
                spin_axis=vector(data["spin_axis_unit"], "spin_axis_unit"),
                ball_mass=positive(data["ball_mass_kg"], "ball_mass_kg", 100.0),
                ball_radius=positive(data["ball_radius_m"], "ball_radius_m", 10.0),
                air_density=positive(
                    data["air_density_kg_m3"], "air_density_kg_m3", 100.0
                ),
                gravity=positive(data["gravity_m_s2"], "gravity_m_s2", 100.0),
                wind_speed=wind_speed,
                wind_direction=finite(data["wind_direction_rad"], "wind_direction_rad"),
                ball_setup=_ball_setup_from_dict(data["ball_setup"]),
            )
        )


@dataclass(frozen=True)
class FlightExecutionInput:
    """Flight model authority, bounded numeric settings, and result identity."""

    model_id: str
    model_version: str
    settings: Mapping[str, float]
    trajectory_sha256: str
    result_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", canonical_text(self.model_id, "model_id"))
        object.__setattr__(
            self, "model_version", canonical_text(self.model_version, "model_version")
        )
        if not isinstance(self.settings, Mapping) or not (
            1 <= len(self.settings) <= MAX_FLIGHT_SETTINGS
        ):
            raise ValueError("flight settings must contain between 1 and 64 values")
        normalized = {
            stable_id(key, "flight setting id"): finite(value, f"flight setting {key}")
            for key, value in self.settings.items()
        }
        object.__setattr__(self, "settings", MappingProxyType(normalized))
        object.__setattr__(
            self,
            "trajectory_sha256",
            digest(self.trajectory_sha256, "trajectory_sha256"),
        )
        object.__setattr__(
            self, "result_sha256", digest(self.result_sha256, "result_sha256")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the detached flight-input mapping."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "settings": dict(self.settings),
            "trajectory_sha256": self.trajectory_sha256,
            "result_sha256": self.result_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> FlightExecutionInput:
        """Parse an exact flight-input mapping."""
        data = exact_mapping(value, _FLIGHT_FIELDS, "flight")
        settings = data["settings"]
        if not isinstance(settings, Mapping):
            raise TypeError("flight settings must be a JSON object")
        return cls(
            data["model_id"],
            data["model_version"],
            settings,
            data["trajectory_sha256"],
            data["result_sha256"],
        )


@dataclass(frozen=True)
class GroundExecutionOptions:
    """Bounded orchestration controls that do not define physics."""

    max_trials: int
    max_parallelism: int
    timeout_s: float
    fail_fast: bool

    def __post_init__(self) -> None:
        integer(self.max_trials, "max_trials", 1, 10_000)
        integer(
            self.max_parallelism,
            "max_parallelism",
            1,
            MAX_EXECUTION_PARALLELISM,
        )
        positive(self.timeout_s, "timeout_s", MAX_EXECUTION_TIMEOUT_S)
        if type(self.fail_fast) is not bool:
            raise TypeError("fail_fast must be a boolean")

    def to_dict(self) -> dict[str, object]:
        """Return the exact execution-options mapping."""
        return {
            "max_trials": self.max_trials,
            "max_parallelism": self.max_parallelism,
            "timeout_s": self.timeout_s,
            "fail_fast": self.fail_fast,
        }

    @classmethod
    def from_dict(cls, value: object) -> GroundExecutionOptions:
        """Parse exact bounded orchestration controls."""
        data = exact_mapping(value, _EXECUTION_FIELDS, "execution_options")
        return cls(
            data["max_trials"],
            data["max_parallelism"],
            data["timeout_s"],
            data["fail_fast"],
        )


def _trajectory_payload(result: FlightResult) -> list[dict[str, object]]:
    if type(result) is not FlightResult:
        raise TypeError("result must be an exact FlightResult")
    return [
        {
            "time_s": point.time,
            "position_m": point.position.tolist(),
            "velocity_m_s": point.velocity.tolist(),
            "angular_velocity_rad_s": point.angular_velocity_rad_s.tolist()
            if isinstance(point, FlightStatePoint)
            else None,
        }
        for point in result.trajectory
    ]


def canonical_flight_trajectory_sha256(result: FlightResult) -> str:
    """Hash every canonical flight trajectory sample and angular state."""
    return sha256(_trajectory_payload(result))


def canonical_flight_result_sha256(result: FlightResult) -> str:
    """Hash the trajectory identity plus every scalar flight-result field."""
    return sha256(
        {
            "trajectory_sha256": canonical_flight_trajectory_sha256(result),
            "model_name": result.model_name,
            "carry_distance_m": result.carry_distance,
            "max_height_m": result.max_height,
            "flight_time_s": result.flight_time,
            "landing_angle_deg": result.landing_angle,
            "lateral_deviation_m": result.lateral_deviation,
        }
    )


__all__ = [
    "MAX_CAPTURE_SPEED_M_S",
    "MAX_EXECUTION_TIMEOUT_S",
    "FlightExecutionInput",
    "FlightLaunchInput",
    "GroundExecutionOptions",
    "canonical_flight_result_sha256",
    "canonical_flight_trajectory_sha256",
    "canonical_text",
    "digest",
    "finite",
    "integer",
    "positive",
    "sha256",
]
