"""Typed non-wire contracts for planar skid/roll surface motion."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

from ._vector_math import dot, norm, subtract
from .contract_types import (
    GroundSurfaceProfile,
    Vector3,
)

GROUND_SKID_ROLL_MODEL_ID = "tools-ground-skid-roll"
GROUND_SKID_ROLL_MODEL_VERSION = "1.1.0"
STANDARD_GRAVITY_M_S2: Vector3 = (0.0, -9.80665, 0.0)
CancellationCheck = Callable[[], bool]


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: float, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _vector(value: Vector3, name: str) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three components")
    return (
        _finite(value[0], name),
        _finite(value[1], name),
        _finite(value[2], name),
    )


@dataclass(frozen=True)
class PlanarSurfaceDomain:
    """One immutable plane, optionally finite along one tangent axis."""

    surface: GroundSurfaceProfile
    axis_origin_m: Vector3 = (0.0, 0.0, 0.0)
    axis_unit: Vector3 = (1.0, 0.0, 0.0)
    lower_coordinate_m: float | None = None
    upper_coordinate_m: float | None = None

    def __post_init__(self) -> None:
        if type(self.surface) is not GroundSurfaceProfile:
            raise ValueError("domain surface must be an exact surface profile")
        origin = _vector(self.axis_origin_m, "axis_origin_m")
        axis = _vector(self.axis_unit, "axis_unit")
        lower = self._optional_bound(self.lower_coordinate_m, "lower coordinate")
        upper = self._optional_bound(self.upper_coordinate_m, "upper coordinate")
        object.__setattr__(self, "axis_origin_m", origin)
        object.__setattr__(self, "axis_unit", axis)
        object.__setattr__(self, "lower_coordinate_m", lower)
        object.__setattr__(self, "upper_coordinate_m", upper)
        self._validate_geometry()
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("domain lower coordinate must be below upper coordinate")

    @staticmethod
    def _optional_bound(value: float | None, name: str) -> float | None:
        return None if value is None else _finite(value, name)

    def _validate_geometry(self) -> None:
        if not math.isclose(norm(self.axis_unit), 1.0, abs_tol=1e-10):
            raise ValueError("domain axis_unit must be a unit vector")
        if abs(dot(self.axis_unit, self.surface.normal_unit)) > 1e-10:
            raise ValueError("domain axis_unit must be tangent to the plane")
        plane_origin = (0.0, self.surface.height_m, 0.0)
        offset = subtract(self.axis_origin_m, plane_origin)
        if abs(dot(offset, self.surface.normal_unit)) > 1e-9:
            raise ValueError("domain axis origin must lie on the plane")

    def coordinate(self, position_m: Vector3) -> float:
        """Return signed position along the finite-domain axis."""
        return float(dot(subtract(position_m, self.axis_origin_m), self.axis_unit))

    def contains(self, position_m: Vector3) -> bool:
        """Return whether a center projection is inside both finite bounds."""
        coordinate = self.coordinate(position_m)
        lower_ok = (
            self.lower_coordinate_m is None or coordinate >= self.lower_coordinate_m
        )
        upper_ok = (
            self.upper_coordinate_m is None or coordinate <= self.upper_coordinate_m
        )
        return lower_ok and upper_ok


@dataclass(frozen=True)
class SurfaceKinematicSegment:
    """Constant-acceleration center path used for exact edge localization."""

    start_position_m: Vector3
    start_velocity_m_s: Vector3
    acceleration_m_s2: Vector3
    duration_s: float

    def __post_init__(self) -> None:
        for name in ("start_position_m", "start_velocity_m_s", "acceleration_m_s2"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        object.__setattr__(self, "duration_s", _positive(self.duration_s, "duration_s"))

    def position_at(self, time_offset_s: float) -> Vector3:
        """Return the exact constant-acceleration position."""
        time = _finite(time_offset_s, "time_offset_s")
        if not 0.0 <= time <= self.duration_s:
            raise ValueError("time_offset_s must lie within the segment")
        return (
            self.start_position_m[0]
            + self.start_velocity_m_s[0] * time
            + 0.5 * self.acceleration_m_s2[0] * time**2,
            self.start_position_m[1]
            + self.start_velocity_m_s[1] * time
            + 0.5 * self.acceleration_m_s2[1] * time**2,
            self.start_position_m[2]
            + self.start_velocity_m_s[2] * time
            + 0.5 * self.acceleration_m_s2[2] * time**2,
        )


@dataclass(frozen=True)
class SurfaceBoundaryCrossing:
    """Exact first finite-domain edge crossing within one motion segment."""

    time_offset_s: float
    position_m: Vector3
    boundary_coordinate_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "time_offset_s", _finite(self.time_offset_s, "time_offset_s")
        )
        object.__setattr__(self, "position_m", _vector(self.position_m, "position_m"))
        object.__setattr__(
            self,
            "boundary_coordinate_m",
            _finite(self.boundary_coordinate_m, "boundary_coordinate_m"),
        )
        if self.time_offset_s < 0.0:
            raise ValueError("boundary crossing time must be nonnegative")


@dataclass(frozen=True)
class RigidMotion:
    """Constant linear/angular acceleration and contact-force evidence."""

    acceleration_m_s2: Vector3
    angular_acceleration_rad_s2: Vector3
    contact_slip_acceleration_m_s2: Vector3
    contact_force_n: Vector3

    def __post_init__(self) -> None:
        for name in (
            "acceleration_m_s2",
            "angular_acceleration_rad_s2",
            "contact_slip_acceleration_m_s2",
            "contact_force_n",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), name))


class SkidRollTerminationReason(StrEnum):
    """Representable and internal-only suffix termination reasons."""

    REST = "rest"
    LEFT_SURFACE = "left_surface"
    TIME_LIMIT = "time_limit"
    EVENT_LIMIT = "event_limit"
    CANCELLED = "cancelled"
    STEP_LIMIT = "step_limit"
    SURFACE_TRANSITION_LIMIT = "surface_transition_limit"
    UNSUPPORTED_SURFACE = "unsupported_surface"
    NUMERICAL_FAILURE = "numerical_failure"


@dataclass(frozen=True)
class SkidRollSettings:
    """Versioned deterministic integration and threshold settings."""

    integration_step_s: float = 0.001
    max_steps: int = 200_000
    max_surface_transitions: int = 1_024
    velocity_tolerance_m_s: float = 1e-9
    angular_tolerance_rad_s: float = 1e-9
    slip_tolerance_m_s: float = 1e-9
    time_tolerance_s: float = 1e-12
    gravity_m_s2: Vector3 = STANDARD_GRAVITY_M_S2
    model_id: str = GROUND_SKID_ROLL_MODEL_ID
    model_version: str = GROUND_SKID_ROLL_MODEL_VERSION

    def __post_init__(self) -> None:
        for name in (
            "integration_step_s",
            "velocity_tolerance_m_s",
            "angular_tolerance_rad_s",
            "slip_tolerance_m_s",
            "time_tolerance_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("max_steps", "max_surface_transitions"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if tuple(self.gravity_m_s2) != STANDARD_GRAVITY_M_S2:
            raise ValueError("gravity_m_s2 must equal versioned standard gravity")
        object.__setattr__(self, "gravity_m_s2", STANDARD_GRAVITY_M_S2)
        if not self.model_id or not self.model_version:
            raise ValueError("skid/roll model identity must be nonempty")


@dataclass(frozen=True)
class SkidRollEnergyLedger:
    """Ground suffix mechanical-energy and moving-surface work ledger."""

    kinetic_before_j: float
    kinetic_after_j: float
    gravity_work_j: float
    surface_work_j: float
    dissipation_j: float

    def __post_init__(self) -> None:
        for name in (
            "kinetic_before_j",
            "kinetic_after_j",
            "gravity_work_j",
            "surface_work_j",
            "dissipation_j",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.kinetic_before_j < 0.0 or self.kinetic_after_j < 0.0:
            raise ValueError("kinetic energy must be nonnegative")
        if self.dissipation_j < 0.0:
            raise ValueError("energy dissipation must be nonnegative")


@dataclass(frozen=True)
class SkidRollTermination:
    """Absolute and ground-elapsed suffix termination evidence."""

    reason: SkidRollTerminationReason
    time_s: float
    elapsed_time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", SkidRollTerminationReason(self.reason))
        object.__setattr__(self, "time_s", _finite(self.time_s, "termination time_s"))
        object.__setattr__(
            self, "elapsed_time_s", _finite(self.elapsed_time_s, "elapsed_time_s")
        )
        if self.time_s < 0.0 or self.elapsed_time_s < 0.0:
            raise ValueError("termination times must be nonnegative")
        if self.elapsed_time_s > self.time_s:
            raise ValueError("elapsed termination time cannot exceed absolute time")


__all__ = [
    "CancellationCheck",
    "GROUND_SKID_ROLL_MODEL_ID",
    "GROUND_SKID_ROLL_MODEL_VERSION",
    "PlanarSurfaceDomain",
    "RigidMotion",
    "STANDARD_GRAVITY_M_S2",
    "SkidRollEnergyLedger",
    "SkidRollSettings",
    "SkidRollTermination",
    "SkidRollTerminationReason",
    "SurfaceBoundaryCrossing",
    "SurfaceKinematicSegment",
]
