"""Validated value types for the flight-to-ground v1 boundary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias

from shared.python.swing_sim.canonical_numeric_json import canonical_numeric_float

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

REQUEST_SCHEMA_VERSION = "flight-to-ground-request/v1"
RESULT_SCHEMA_VERSION = "flight-to-ground-result/v1"
UNIT_SYSTEM_SI = "SI"
Vector3: TypeAlias = tuple[float, float, float]
_UNIT_TOLERANCE = 1e-9
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_MIN_CANONICAL_POSITIVE = 0.00000000001
_TEXT_EDGE_WHITESPACE = " \t\r\n\f\v"


def _raw_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _finite(value: float, name: str) -> float:
    return float(canonical_numeric_float(_raw_finite(value, name)))


def _nonnegative(value: float, name: str) -> float:
    raw = _raw_finite(value, name)
    if raw < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return float(canonical_numeric_float(raw))


def _bounded(value: float, name: str, upper: float = 1.0) -> float:
    raw = _raw_finite(value, name)
    if not 0.0 <= raw <= upper:
        raise ValueError(f"{name} must lie within [0, {upper:g}]")
    return float(canonical_numeric_float(raw))


def _positive(value: float, name: str) -> float:
    raw = _raw_finite(value, name)
    if raw < _MIN_CANONICAL_POSITIVE:
        raise ValueError(f"{name} must be at least {_MIN_CANONICAL_POSITIVE:g}")
    number = float(canonical_numeric_float(raw))
    return number


def _integer(value: int | float, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be an integer")
    if not minimum <= normalized <= _MAX_SAFE_INTEGER:
        raise ValueError(
            f"{name} must lie within cross-runtime safe range "
            f"[{minimum}, {_MAX_SAFE_INTEGER}]"
        )
    return normalized


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty")
    if value != value.strip(_TEXT_EDGE_WHITESPACE):
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} must not contain surrogate code points")
    return value


def _vector(value: Vector3, name: str) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three components")
    return tuple(_finite(component, name) for component in value)  # type: ignore[return-value]


class _WireRecord:
    """Delegate strict serialization without coupling records to parser details."""

    def to_dict(self) -> dict[str, Any]:
        """Return the strict v1 JSON-compatible mapping."""
        from .contract_wire import record_to_dict

        payload: dict[str, Any] = record_to_dict(self)
        return payload

    def to_json(self) -> str:
        """Return deterministic compact JSON."""
        from shared.python.swing_sim.canonical_numeric_json import (
            canonical_numeric_json,
        )

        return str(canonical_numeric_json(self.to_dict()))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Any:
        """Parse a strict v1 mapping for this record type."""
        from .contract_wire import record_from_dict

        return record_from_dict(cls, payload)


class GroundFrame(StrEnum):
    """Frame supported by v1; no implicit frame conversion is permitted."""

    TARGET = "target_frame:x_downrange,y_up,z_right"


class GroundPhase(StrEnum):
    """Physical phases represented in a ground-run trajectory."""

    IMPACT = "impact"
    BOUNCE = "bounce"
    SKID = "skid"
    ROLL = "roll"
    REST = "rest"


class GroundEventType(StrEnum):
    """Discrete transitions recorded by a ground solver."""

    FIRST_CONTACT = "first_contact"
    BOUNCE = "bounce"
    SKID_TO_ROLL = "skid_to_roll"
    SURFACE_TRANSITION = "surface_transition"
    REST = "rest"
    LEFT_SURFACE = "left_surface"


class CalibrationKind(StrEnum):
    """Evidence class for material and solver calibration."""

    MEASURED = "measured"
    LITERATURE = "literature"
    ESTIMATED = "estimated"
    UNVALIDATED = "unvalidated"


class GroundWarningSeverity(StrEnum):
    """Typed severity for non-fatal model qualifications."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class GroundTerminationReason(StrEnum):
    """Typed solver termination reasons."""

    REST = "rest"
    TIME_LIMIT = "time_limit"
    EVENT_LIMIT = "event_limit"
    LEFT_SURFACE = "left_surface"
    NUMERICAL_FAILURE = "numerical_failure"
    UNAVAILABLE_INPUT = "unavailable_input"


class GroundResultStatus(StrEnum):
    """Availability and completion status for a ground-run result."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class GroundProvenance(_WireRecord):
    """Stable producer and input identity for reproducible handoff records."""

    producer: str
    producer_version: str
    source_revision: str
    input_sha256: str

    def __post_init__(self) -> None:
        for name in ("producer", "producer_version", "source_revision"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        digest = _text(self.input_sha256, "input_sha256").lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("input_sha256 must be 64 lowercase hexadecimal characters")
        object.__setattr__(self, "input_sha256", digest)


@dataclass(frozen=True)
class GroundCalibration(_WireRecord):
    """Calibration evidence attached to both request and result."""

    calibration_id: str
    kind: CalibrationKind
    source: str
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "calibration_id", _text(self.calibration_id, "calibration_id")
        )
        object.__setattr__(self, "kind", CalibrationKind(self.kind))
        object.__setattr__(self, "source", _text(self.source, "calibration source"))
        object.__setattr__(self, "confidence", _bounded(self.confidence, "confidence"))


@dataclass(frozen=True)
class GroundSurfaceProfile(_WireRecord):
    """Planar SI material profile in the canonical target frame."""

    surface_id: str
    provider_id: str
    provider_version: str
    frame: GroundFrame
    height_m: float
    normal_unit: Vector3
    surface_velocity_m_s: Vector3
    normal_restitution: float
    static_friction: float
    kinetic_friction: float
    rolling_resistance: float
    firmness_pa: float
    hardness_fraction: float
    grass_height_m: float
    compressibility_fraction: float
    compression_damping_fraction: float
    turf_density_kg_m3: float
    moisture_fraction: float

    def __post_init__(self) -> None:
        static_raw = _raw_finite(self.static_friction, "static_friction")
        kinetic_raw = _raw_finite(self.kinetic_friction, "kinetic_friction")
        if kinetic_raw > static_raw:
            raise ValueError("kinetic_friction must not exceed static_friction")
        self._normalize_identity()
        self._normalize_geometry()
        self._normalize_material()

    def _normalize_identity(self) -> None:
        object.__setattr__(self, "surface_id", _text(self.surface_id, "surface_id"))
        object.__setattr__(self, "provider_id", _text(self.provider_id, "provider_id"))
        object.__setattr__(
            self, "provider_version", _text(self.provider_version, "provider_version")
        )
        object.__setattr__(self, "frame", GroundFrame(self.frame))

    def _normalize_geometry(self) -> None:
        object.__setattr__(self, "height_m", _finite(self.height_m, "height_m"))
        normal = _vector(self.normal_unit, "normal_unit")
        if (
            abs(math.sqrt(sum(value * value for value in normal)) - 1.0)
            > _UNIT_TOLERANCE
        ):
            raise ValueError("normal_unit must be a unit vector")
        if normal[1] <= 0.0:
            raise ValueError("normal_unit must point upward in the target frame")
        object.__setattr__(self, "normal_unit", normal)
        object.__setattr__(
            self,
            "surface_velocity_m_s",
            _vector(self.surface_velocity_m_s, "surface_velocity_m_s"),
        )
        normal_speed = sum(
            self.surface_velocity_m_s[index] * self.normal_unit[index]
            for index in range(3)
        )
        if abs(normal_speed) > _UNIT_TOLERANCE:
            raise ValueError("v1 surface_velocity_m_s must be tangential to the plane")

    def _normalize_material(self) -> None:
        self._set_bounded("normal_restitution")
        self._set_bounded("static_friction", 5.0)
        self._set_bounded("kinetic_friction", 5.0)
        self._set_bounded("rolling_resistance")
        object.__setattr__(
            self, "firmness_pa", _positive(self.firmness_pa, "firmness_pa")
        )
        self._set_bounded("hardness_fraction")
        object.__setattr__(
            self, "grass_height_m", _nonnegative(self.grass_height_m, "grass_height_m")
        )
        self._set_bounded("compressibility_fraction")
        self._set_bounded("compression_damping_fraction")
        object.__setattr__(
            self,
            "turf_density_kg_m3",
            _nonnegative(self.turf_density_kg_m3, "turf_density_kg_m3"),
        )
        self._set_bounded("moisture_fraction")

    def _set_bounded(self, name: str, upper: float = 1.0) -> None:
        object.__setattr__(self, name, _bounded(getattr(self, name), name, upper))

    def signed_gap_m(self, state: GroundContactState, radius_m: float) -> float:
        """Return sphere-to-plane signed gap; positive means separated."""
        if state.frame is not self.frame:
            raise ValueError("contact frame must match surface frame")
        plane_origin = (0.0, self.height_m, 0.0)
        offset = tuple(state.position_m[i] - plane_origin[i] for i in range(3))
        return sum(offset[i] * self.normal_unit[i] for i in range(3)) - radius_m

    def relative_normal_speed_m_s(self, state: GroundContactState) -> float:
        """Return ball velocity relative to the surface along the plane normal."""
        if state.frame is not self.frame:
            raise ValueError("contact frame must match surface frame")
        relative = tuple(
            state.velocity_m_s[index] - self.surface_velocity_m_s[index]
            for index in range(3)
        )
        return sum(relative[index] * self.normal_unit[index] for index in range(3))


@dataclass(frozen=True)
class GroundContactState(_WireRecord):
    """One full flight state used to bracket physical sphere contact."""

    time_s: float
    frame: GroundFrame
    position_m: Vector3
    velocity_m_s: Vector3
    angular_velocity_rad_s: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _nonnegative(self.time_s, "time_s"))
        object.__setattr__(self, "frame", GroundFrame(self.frame))
        for name in ("position_m", "velocity_m_s", "angular_velocity_rad_s"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))


@dataclass(frozen=True)
class GroundTrajectoryPoint(GroundContactState):
    """One ordered ground-run sample with an explicit physical phase."""

    phase: GroundPhase

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "phase", GroundPhase(self.phase))
        if self.phase is GroundPhase.REST:
            moving = any(abs(value) > 1e-9 for value in self.velocity_m_s)
            spinning = any(abs(value) > 1e-9 for value in self.angular_velocity_rad_s)
            if moving or spinning:
                raise ValueError("rest phase requires zero linear and angular velocity")


@dataclass(frozen=True)
class GroundEvent(_WireRecord):
    """One ordered discontinuity with exact linear and angular states."""

    sequence: int
    event_type: GroundEventType
    time_s: float
    frame: GroundFrame
    position_m: Vector3
    velocity_before_m_s: Vector3
    velocity_after_m_s: Vector3
    angular_velocity_before_rad_s: Vector3
    angular_velocity_after_rad_s: Vector3

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", _integer(self.sequence, "event sequence"))
        object.__setattr__(self, "event_type", GroundEventType(self.event_type))
        object.__setattr__(self, "time_s", _nonnegative(self.time_s, "event time_s"))
        object.__setattr__(self, "frame", GroundFrame(self.frame))
        for name in (
            "position_m",
            "velocity_before_m_s",
            "velocity_after_m_s",
            "angular_velocity_before_rad_s",
            "angular_velocity_after_rad_s",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        if self.event_type is GroundEventType.REST:
            linear = any(abs(value) > 1e-9 for value in self.velocity_after_m_s)
            angular = any(
                abs(value) > 1e-9 for value in self.angular_velocity_after_rad_s
            )
            if linear or angular:
                raise ValueError("rest event requires zero output velocity and spin")
