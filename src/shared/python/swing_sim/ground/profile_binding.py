"""Bind versioned ground material profiles to explicit solver geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .contract_types import GroundFrame, GroundSurfaceProfile, Vector3
from .profile_types import (
    GroundApplicability,
    GroundMaterialProfile,
    GroundModelUseStatus,
    GroundProfileQualification,
    GroundQualificationStatus,
)
from .profile_validation import (
    bounded_number,
    exact_record,
    finite_number,
    positive_number,
    sha256_digest,
    strict_text,
)

PROFILE_UNQUALIFIED_WARNING = "GROUND_PROFILE_UNQUALIFIED"
PROFILE_ILLUSTRATIVE_WARNING = "GROUND_PROFILE_ILLUSTRATIVE"
_UNIT_TOLERANCE = 1e-10


def _vector(value: Vector3, name: str) -> Vector3:
    if len(value) != 3:
        raise ValueError(f"{name} must contain three components")
    return (
        finite_number(value[0], name),
        finite_number(value[1], name),
        finite_number(value[2], name),
    )


@dataclass(frozen=True)
class SurfacePlacement:
    """Explicit plane geometry for one material-profile binding."""

    surface_id: str
    height_m: float
    normal_unit: Vector3
    surface_velocity_m_s: Vector3
    frame: GroundFrame = GroundFrame.TARGET

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "surface_id", strict_text(self.surface_id, "surface_id")
        )
        object.__setattr__(self, "height_m", finite_number(self.height_m, "height_m"))
        normal = _vector(self.normal_unit, "normal_unit")
        velocity = _vector(self.surface_velocity_m_s, "surface_velocity_m_s")
        if abs(math.hypot(*normal) - 1.0) > _UNIT_TOLERANCE:
            raise ValueError("normal_unit must be a unit vector")
        if normal[1] <= 0.0:
            raise ValueError("normal_unit must point upward")
        if (
            abs(sum(a * b for a, b in zip(normal, velocity, strict=True)))
            > _UNIT_TOLERANCE
        ):
            raise ValueError("surface_velocity_m_s must be tangential")
        object.__setattr__(self, "normal_unit", normal)
        object.__setattr__(self, "surface_velocity_m_s", velocity)
        object.__setattr__(self, "frame", GroundFrame(self.frame))


@dataclass(frozen=True)
class ProfileOperatingCondition:
    """Ambient state required to evaluate profile applicability."""

    surface_class: str
    temperature_k: float
    moisture_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "surface_class", strict_text(self.surface_class, "surface_class")
        )
        object.__setattr__(
            self, "temperature_k", positive_number(self.temperature_k, "temperature_k")
        )
        object.__setattr__(
            self,
            "moisture_fraction",
            bounded_number(self.moisture_fraction, "moisture_fraction", (0.0, 1.0)),
        )


@dataclass(frozen=True)
class BoundGroundSurface:
    """Solver surface plus immutable source profile qualification evidence."""

    surface: GroundSurfaceProfile
    profile: GroundMaterialProfile
    profile_id: str
    profile_revision: str
    profile_sha256: str
    qualification: GroundProfileQualification
    applicability: GroundApplicability
    operating_condition: ProfileOperatingCondition
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.surface) is not GroundSurfaceProfile:
            raise TypeError("surface must use the exact solver contract type")
        exact_record(self.profile, GroundMaterialProfile, "profile")
        for name in ("profile_id", "profile_revision"):
            object.__setattr__(self, name, strict_text(getattr(self, name), name))
        digest = sha256_digest(self.profile_sha256, "profile_sha256")
        if self.profile_id != self.profile.profile_id:
            raise ValueError("profile_id does not match profile")
        if self.profile_revision != self.profile.revision:
            raise ValueError("profile_revision does not match profile")
        if digest != self.profile.canonical_sha256():
            raise ValueError("profile_sha256 does not match profile")
        exact_record(self.qualification, GroundProfileQualification, "qualification")
        exact_record(self.applicability, GroundApplicability, "applicability")
        exact_record(
            self.operating_condition,
            ProfileOperatingCondition,
            "operating_condition",
        )
        if self.qualification != self.profile.qualification:
            raise ValueError("qualification does not match profile")
        if self.applicability != self.profile.applicability:
            raise ValueError("applicability does not match profile")
        _validate_operating_condition(self.applicability, self.operating_condition)
        if (
            self.surface.provider_id != self.profile.provenance.producer
            or self.surface.provider_version != self.profile.provenance.producer_version
        ):
            raise ValueError("surface provider does not match profile provenance")
        if _surface_material_values(self.surface) != tuple(
            item.value_si for item in self.profile.parameters
        ):
            raise ValueError("surface material values do not match profile")
        expected_warnings = _profile_warnings(self.profile)
        if type(self.warnings) is not tuple or self.warnings != expected_warnings:
            raise ValueError("warnings do not match profile qualification")


def _validate_binding_inputs(
    profile: GroundMaterialProfile,
    placement: SurfacePlacement,
    operating_condition: ProfileOperatingCondition,
) -> GroundApplicability:
    if type(profile) is not GroundMaterialProfile:
        raise TypeError("profile must be an exact GroundMaterialProfile")
    if type(placement) is not SurfacePlacement:
        raise TypeError("placement must be an exact SurfacePlacement")
    if type(operating_condition) is not ProfileOperatingCondition:
        raise TypeError("operating_condition must use the exact contract type")
    applicability = profile.applicability
    _validate_operating_condition(applicability, operating_condition)
    return applicability


def _validate_operating_condition(
    applicability: GroundApplicability,
    operating_condition: ProfileOperatingCondition,
) -> None:
    is_applicable = (
        operating_condition.surface_class in applicability.surface_classes
        and applicability.temperature_min_k
        <= operating_condition.temperature_k
        <= applicability.temperature_max_k
        and applicability.moisture_min_fraction
        <= operating_condition.moisture_fraction
        <= applicability.moisture_max_fraction
    )
    if not is_applicable:
        raise ValueError("operating condition lies outside profile applicability")


def _surface_material_values(surface: GroundSurfaceProfile) -> tuple[float, ...]:
    return (
        surface.normal_restitution,
        surface.static_friction,
        surface.kinetic_friction,
        surface.rolling_resistance,
        surface.firmness_pa,
        surface.hardness_fraction,
        surface.grass_height_m,
        surface.compressibility_fraction,
        surface.compression_damping_fraction,
        surface.turf_density_kg_m3,
        surface.moisture_fraction,
    )


def _surface_from_profile(
    profile: GroundMaterialProfile,
    placement: SurfacePlacement,
) -> GroundSurfaceProfile:
    values = {str(item.parameter_id): item.value_si for item in profile.parameters}
    return GroundSurfaceProfile(
        placement.surface_id,
        profile.provenance.producer,
        profile.provenance.producer_version,
        placement.frame,
        placement.height_m,
        placement.normal_unit,
        placement.surface_velocity_m_s,
        values["normal_restitution"],
        values["static_friction"],
        values["kinetic_friction"],
        values["rolling_resistance"],
        values["firmness_pa"],
        values["hardness_fraction"],
        values["grass_height_m"],
        values["compressibility_fraction"],
        values["compression_damping_fraction"],
        values["turf_density_kg_m3"],
        values["moisture_fraction"],
    )


def _profile_warnings(profile: GroundMaterialProfile) -> tuple[str, ...]:
    warnings: list[str] = []
    if profile.qualification.status is GroundQualificationStatus.UNQUALIFIED:
        warnings.append(PROFILE_UNQUALIFIED_WARNING)
    if profile.model_use_status is GroundModelUseStatus.ILLUSTRATIVE:
        warnings.append(PROFILE_ILLUSTRATIVE_WARNING)
    return tuple(warnings)


def bind_material_profile(
    profile: GroundMaterialProfile,
    placement: SurfacePlacement,
    operating_condition: ProfileOperatingCondition,
) -> BoundGroundSurface:
    """Map all eleven profile values to one explicit target-frame plane."""
    applicability = _validate_binding_inputs(profile, placement, operating_condition)
    surface = _surface_from_profile(profile, placement)
    warnings = _profile_warnings(profile)
    return BoundGroundSurface(
        surface,
        profile,
        profile.profile_id,
        profile.revision,
        profile.canonical_sha256(),
        profile.qualification,
        applicability,
        operating_condition,
        warnings,
    )


__all__ = [
    "BoundGroundSurface",
    "PROFILE_ILLUSTRATIVE_WARNING",
    "PROFILE_UNQUALIFIED_WARNING",
    "ProfileOperatingCondition",
    "SurfacePlacement",
    "bind_material_profile",
]
