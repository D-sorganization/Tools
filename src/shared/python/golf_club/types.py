"""Immutable SI and frame-explicit golf-club component contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from ._validation import (
    Matrix3,
    Vector3,
    require_finite_float,
    require_identifier,
    require_inertia,
    require_rotation,
    require_vector3,
)

_IDENTITY: Matrix3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)


class ComponentRole(StrEnum):
    """Stable physical role of one club assembly component."""

    HEAD = "head"
    SHAFT = "shaft"
    GRIP = "grip"
    ADAPTER = "adapter"
    FERRULE = "ferrule"
    ADDED_WEIGHT = "added_weight"


class ClubLengthConvention(StrEnum):
    """Declared reference convention used by a length measurement.

    The record stores a measured value and its datum identifiers. It does not
    infer geometry or certify compliance with any governing body's rules.
    """

    DECLARED_DATUMS = "declared_datums"
    SIXTY_DEGREE_SOLE_PLANE = "sixty_degree_sole_plane"


@dataclass(frozen=True)
class RigidTransform:
    """Proper rigid transform mapping ``from_frame_id`` into ``to_frame_id``.

    Point convention: ``p_to = rotation @ p_from + translation_m``. Rotation
    is dimensionless; translation and transformed points are in metres.
    """

    from_frame_id: str
    to_frame_id: str
    rotation: Matrix3 = _IDENTITY
    translation_m: Vector3 = _ZERO_VECTOR

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "from_frame_id",
            require_identifier(self.from_frame_id, "from_frame_id"),
        )
        object.__setattr__(
            self, "to_frame_id", require_identifier(self.to_frame_id, "to_frame_id")
        )
        object.__setattr__(self, "rotation", require_rotation(self.rotation))
        object.__setattr__(
            self,
            "translation_m",
            require_vector3(self.translation_m, "translation_m"),
        )

    def transform_point(self, point_m: object) -> Vector3:
        """Map one point in metres into ``to_frame_id``."""
        point = require_vector3(point_m, "point_m")
        mapped = np.asarray(self.rotation) @ np.asarray(point)
        mapped += np.asarray(self.translation_m)
        return tuple(float(value) for value in mapped)  # type: ignore[return-value]

    def rotate_inertia_tensor(self, tensor: object) -> Matrix3:
        """Rotate a symmetric positive-semidefinite inertia tensor."""
        source = require_inertia(tensor)
        rotation = np.asarray(self.rotation)
        rotated = rotation @ np.asarray(source) @ rotation.T
        return require_inertia(rotated)


@dataclass(frozen=True)
class ComponentMassProperties:
    """Mass, local center of mass, and inertia about that center in SI units."""

    component_id: str
    role: ComponentRole
    frame_id: str
    mass_kg: float
    center_of_mass_m: Vector3
    inertia_at_com_kg_m2: Matrix3

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "component_id", require_identifier(self.component_id, "component_id")
        )
        if not isinstance(self.role, ComponentRole):
            raise TypeError("role must be a ComponentRole")
        object.__setattr__(
            self, "frame_id", require_identifier(self.frame_id, "frame_id")
        )
        object.__setattr__(
            self,
            "mass_kg",
            require_finite_float(self.mass_kg, "mass_kg", positive=True),
        )
        object.__setattr__(
            self,
            "center_of_mass_m",
            require_vector3(self.center_of_mass_m, "center_of_mass_m"),
        )
        object.__setattr__(
            self,
            "inertia_at_com_kg_m2",
            require_inertia(self.inertia_at_com_kg_m2),
        )


@dataclass(frozen=True)
class ClubComponent:
    """One validated component placed into a club assembly frame."""

    mass_properties: ComponentMassProperties
    transform_to_club: RigidTransform

    def __post_init__(self) -> None:
        if not isinstance(self.mass_properties, ComponentMassProperties):
            raise TypeError("mass_properties must be ComponentMassProperties")
        if not isinstance(self.transform_to_club, RigidTransform):
            raise TypeError("transform_to_club must be RigidTransform")
        if self.mass_properties.frame_id != self.transform_to_club.from_frame_id:
            raise ValueError("component frame_id must match transform from_frame_id")

    @property
    def component_id(self) -> str:
        """Return the component's stable identifier."""
        return self.mass_properties.component_id


@dataclass(frozen=True)
class ClubLengthMeasurement:
    """Measured club length with explicit frame and endpoint references.

    ``lower_reference_id`` and ``upper_reference_id`` identify the two declared
    datums. Geometry engines may later resolve those identifiers; this domain
    slice intentionally stores provenance without calculating club length.
    """

    length_m: float
    convention: ClubLengthConvention
    measurement_frame_id: str
    lower_reference_id: str
    upper_reference_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "length_m",
            require_finite_float(self.length_m, "length_m", positive=True),
        )
        if not isinstance(self.convention, ClubLengthConvention):
            raise TypeError("convention must be a ClubLengthConvention")
        for name in (
            "measurement_frame_id",
            "lower_reference_id",
            "upper_reference_id",
        ):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )


@dataclass(frozen=True)
class AssembledMassProperties:
    """Combined mass properties about the assembly center of mass."""

    frame_id: str
    component_ids: tuple[str, ...]
    total_mass_kg: float
    center_of_mass_m: Vector3
    inertia_at_com_kg_m2: Matrix3

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", require_identifier(self.frame_id, "frame_id")
        )
        if not isinstance(self.component_ids, tuple):
            raise TypeError("component_ids must be a tuple of strings")
        identifiers = tuple(
            require_identifier(value, "component_ids entry")
            for value in self.component_ids
        )
        if not identifiers:
            raise ValueError("component_ids must contain at least one identifier")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("component_ids must be unique")
        object.__setattr__(self, "component_ids", identifiers)
        object.__setattr__(
            self,
            "total_mass_kg",
            require_finite_float(self.total_mass_kg, "total_mass_kg", positive=True),
        )
        object.__setattr__(
            self,
            "center_of_mass_m",
            require_vector3(self.center_of_mass_m, "center_of_mass_m"),
        )
        object.__setattr__(
            self,
            "inertia_at_com_kg_m2",
            require_inertia(self.inertia_at_com_kg_m2),
        )


__all__ = [
    "AssembledMassProperties",
    "ClubComponent",
    "ClubLengthConvention",
    "ClubLengthMeasurement",
    "ComponentMassProperties",
    "ComponentRole",
    "RigidTransform",
]
