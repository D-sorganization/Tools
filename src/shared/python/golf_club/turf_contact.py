"""Passive reduced-order compliant-turf contact for wedge sole diagnostics.

The model is deliberately replaceable and modest in claim scope.  It treats
one declared contact point as a unilateral Kelvin-Voigt normal element with a
smooth Coulomb tangential law.  It can supply a frame-explicit wrench to a
rigid-body integrator or run a reduced effective-mass impact for convergence
and sensitivity studies.  It does not predict divot shape, grass fracture, or
granular sand flow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ._validation import (
    Vector3,
    require_finite_float,
    require_identifier,
    require_vector3,
)

_UNIT_TOLERANCE = 1e-9
_CONTACT_TOLERANCE_M = 1e-12
_MAX_STIFFNESS_N_M = 5_000_000.0
_MAX_DAMPING_N_S_M = 100_000.0
_MAX_FRICTION = 1.0
_MAX_PENETRATION_M = 0.25
_LIMITATIONS = (
    "Reduced single-point Kelvin-Voigt/Coulomb proxy only; no divot shape, "
    "grass fracture, granular flow, injury, or named-course prediction."
)


class TurfCalibrationStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Evidence state controlling which downstream claims are permitted."""

    UNCALIBRATED = "uncalibrated"
    ILLUSTRATIVE = "illustrative"
    CALIBRATED = "calibrated"


class TurfPreset(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Generic starting profiles; none represents a named playing surface."""

    FIRM_FAIRWAY = "firm_fairway"
    SOFT_TURF = "soft_turf"
    SAND_LIKE = "sand_like"


class TurfContactStatus(str, Enum):  # noqa: UP042 - Python 3.10 compatibility
    """Explicit instantaneous or reduced-simulation state."""

    NO_CONTACT = "no_contact"
    NO_RESPONSE = "no_response"
    ACTIVE = "active"
    OUTSIDE_CALIBRATED_DOMAIN = "outside_calibrated_domain"
    SEPARATED = "separated"
    STEP_LIMIT = "step_limit"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TurfProfileProvenance:
    """Visible evidence and uncertainty boundary for surface parameters."""

    source_name: str
    parameter_basis: str
    uncertainty_note: str
    source_uri: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_name", "parameter_basis", "uncertainty_note"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )
        if self.source_uri is not None:
            object.__setattr__(
                self,
                "source_uri",
                require_identifier(self.source_uri, "source_uri"),
            )


@dataclass(frozen=True)
class TurfContactProfile:
    """Versionable SI parameters for the reduced contact law."""

    profile_id: str
    normal_stiffness_n_m: float
    normal_damping_n_s_m: float
    friction_coefficient: float
    friction_regularization_mps: float
    max_penetration_m: float
    calibration_status: TurfCalibrationStatus
    provenance: TurfProfileProvenance

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_id", require_identifier(self.profile_id, "profile_id")
        )
        if not isinstance(self.calibration_status, TurfCalibrationStatus):
            raise TypeError("calibration_status must be TurfCalibrationStatus")
        if not isinstance(self.provenance, TurfProfileProvenance):
            raise TypeError("provenance must be TurfProfileProvenance")
        self._bounded("normal_stiffness_n_m", 0.0, _MAX_STIFFNESS_N_M)
        self._bounded("normal_damping_n_s_m", 0.0, _MAX_DAMPING_N_S_M)
        self._bounded("friction_coefficient", 0.0, _MAX_FRICTION)
        regularization = require_finite_float(
            self.friction_regularization_mps,
            "friction_regularization_mps",
            positive=True,
        )
        object.__setattr__(self, "friction_regularization_mps", regularization)
        penetration = require_finite_float(
            self.max_penetration_m, "max_penetration_m", positive=True
        )
        if penetration > _MAX_PENETRATION_M:
            raise ValueError(f"max_penetration_m must be <= {_MAX_PENETRATION_M}")
        object.__setattr__(self, "max_penetration_m", penetration)

    def _bounded(self, name: str, lower: float, upper: float) -> None:
        value = require_finite_float(getattr(self, name), name)
        if value < lower or value > upper:
            raise ValueError(f"{name} must be in [{lower}, {upper}]")
        object.__setattr__(self, name, value)

    @property
    def supports_turf_rankings(self) -> bool:
        """Whether calibrated-turf ranking language is permitted."""
        return self.calibration_status is TurfCalibrationStatus.CALIBRATED


@dataclass(frozen=True)
class TurfContactKinematics:
    """One contact point and surface state in a declared inertial frame."""

    frame_id: str
    reference_point_m: Vector3
    application_point_m: Vector3
    surface_normal_unit: Vector3
    surface_velocity_mps: Vector3
    contact_point_velocity_mps: Vector3
    penetration_m: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", require_identifier(self.frame_id, "frame_id")
        )
        for name in (
            "reference_point_m",
            "application_point_m",
            "surface_velocity_mps",
            "contact_point_velocity_mps",
        ):
            object.__setattr__(self, name, require_vector3(getattr(self, name), name))
        normal = require_vector3(self.surface_normal_unit, "surface_normal_unit")
        if not math.isclose(
            float(np.linalg.norm(normal)), 1.0, abs_tol=_UNIT_TOLERANCE, rel_tol=0.0
        ):
            raise ValueError("surface_normal_unit must be unit length")
        object.__setattr__(self, "surface_normal_unit", normal)
        penetration = require_finite_float(self.penetration_m, "penetration_m")
        if penetration < 0.0:
            raise ValueError("penetration_m must be >= 0")
        object.__setattr__(self, "penetration_m", penetration)


@dataclass(frozen=True)
class TurfContactResponse:
    """Passive force/wrench and energy terms for one contact evaluation."""

    status: TurfContactStatus
    force_world_n: Vector3
    tangential_force_world_n: Vector3
    torque_at_reference_n_m: Vector3
    normal_force_n: float
    normal_velocity_mps: float
    tangential_speed_mps: float
    stored_elastic_energy_j: float
    dissipated_power_w: float
    effective_penetration_m: float
    limitations: str = _LIMITATIONS


def turf_profile_preset(preset: TurfPreset) -> TurfContactProfile:
    """Return an explicitly illustrative generic surface starting point."""
    if not isinstance(preset, TurfPreset):
        raise TypeError("preset must be TurfPreset")
    values = {
        TurfPreset.FIRM_FAIRWAY: (60_000.0, 220.0, 0.35, 0.025),
        TurfPreset.SOFT_TURF: (20_000.0, 180.0, 0.45, 0.060),
        TurfPreset.SAND_LIKE: (8_000.0, 140.0, 0.55, 0.100),
    }
    stiffness, damping, friction, max_penetration = values[preset]
    label = preset.value.replace("_", " ")
    return TurfContactProfile(
        profile_id=f"illustrative-{preset.value.replace('_', '-')}",
        normal_stiffness_n_m=stiffness,
        normal_damping_n_s_m=damping,
        friction_coefficient=friction,
        friction_regularization_mps=0.02,
        max_penetration_m=max_penetration,
        calibration_status=TurfCalibrationStatus.ILLUSTRATIVE,
        provenance=TurfProfileProvenance(
            source_name=f"illustrative generic {label} proxy",
            parameter_basis=(
                "Engineering starting values for software validation and "
                "user adjustment; not fitted measurements"
            ),
            uncertainty_note=(
                "Uncalibrated reduced-order parameters; do not use for named-course, "
                "divot, injury, or universal bounce-forgiveness claims."
            ),
        ),
    )


def _tuple3(values: np.ndarray) -> Vector3:
    return (float(values[0]), float(values[1]), float(values[2]))


def _zero_response(
    status: TurfContactStatus,
    state: TurfContactKinematics,
    normal_velocity: float,
) -> TurfContactResponse:
    return TurfContactResponse(
        status=status,
        force_world_n=(0.0, 0.0, 0.0),
        tangential_force_world_n=(0.0, 0.0, 0.0),
        torque_at_reference_n_m=(0.0, 0.0, 0.0),
        normal_force_n=0.0,
        normal_velocity_mps=normal_velocity,
        tangential_speed_mps=0.0,
        stored_elastic_energy_j=0.0,
        dissipated_power_w=0.0,
        effective_penetration_m=state.penetration_m,
    )


def evaluate_turf_contact(
    profile: TurfContactProfile,
    state: TurfContactKinematics,
) -> TurfContactResponse:
    """Evaluate a unilateral passive contact wrench at one instant."""
    if not isinstance(profile, TurfContactProfile):
        raise TypeError("profile must be TurfContactProfile")
    if not isinstance(state, TurfContactKinematics):
        raise TypeError("state must be TurfContactKinematics")
    normal = np.asarray(state.surface_normal_unit)
    relative_velocity = np.asarray(state.contact_point_velocity_mps) - np.asarray(
        state.surface_velocity_mps
    )
    normal_velocity = float(relative_velocity @ normal)
    penetration_rate = -normal_velocity
    if profile.normal_stiffness_n_m == profile.normal_damping_n_s_m == 0.0:
        return _zero_response(TurfContactStatus.NO_RESPONSE, state, normal_velocity)
    if state.penetration_m <= _CONTACT_TOLERANCE_M and penetration_rate <= 0.0:
        return _zero_response(TurfContactStatus.NO_CONTACT, state, normal_velocity)
    effective_penetration = min(state.penetration_m, profile.max_penetration_m)
    status = (
        TurfContactStatus.OUTSIDE_CALIBRATED_DOMAIN
        if state.penetration_m > profile.max_penetration_m
        else TurfContactStatus.ACTIVE
    )
    normal_force = max(
        0.0,
        profile.normal_stiffness_n_m * effective_penetration
        + profile.normal_damping_n_s_m * penetration_rate,
    )
    if normal_force == 0.0:
        return _zero_response(TurfContactStatus.NO_CONTACT, state, normal_velocity)
    tangential_velocity = relative_velocity - normal_velocity * normal
    tangential_speed = float(np.linalg.norm(tangential_velocity))
    denominator = math.sqrt(
        tangential_speed**2 + profile.friction_regularization_mps**2
    )
    tangential_force = (
        -profile.friction_coefficient * normal_force * tangential_velocity / denominator
    )
    force = normal_force * normal + tangential_force
    lever = np.asarray(state.application_point_m) - np.asarray(state.reference_point_m)
    torque = np.cross(lever, force)
    damping_power = profile.normal_damping_n_s_m * penetration_rate**2
    friction_power = -float(tangential_force @ tangential_velocity)
    return TurfContactResponse(
        status=status,
        force_world_n=_tuple3(force),
        tangential_force_world_n=_tuple3(tangential_force),
        torque_at_reference_n_m=_tuple3(torque),
        normal_force_n=normal_force,
        normal_velocity_mps=normal_velocity,
        tangential_speed_mps=tangential_speed,
        stored_elastic_energy_j=(
            0.5 * profile.normal_stiffness_n_m * effective_penetration**2
        ),
        dissipated_power_w=damping_power + friction_power,
        effective_penetration_m=effective_penetration,
    )


__all__ = [
    "TurfCalibrationStatus",
    "TurfContactKinematics",
    "TurfContactProfile",
    "TurfContactResponse",
    "TurfContactStatus",
    "TurfPreset",
    "TurfProfileProvenance",
    "evaluate_turf_contact",
    "turf_profile_preset",
]
