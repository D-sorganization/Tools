"""Rigid assembly of component mass properties in a declared club frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from shared.python.model_generation.inertia.primitives import parallel_axis

from ._validation import Matrix3, require_identifier, require_inertia
from .types import (
    AssembledMassProperties,
    ClubComponent,
    ClubLengthMeasurement,
)

_INERTIA_KEYS = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")


def assemble_mass_properties(
    components: Sequence[ClubComponent], assembly_frame_id: str
) -> AssembledMassProperties:
    """Combine rigid component properties in ``assembly_frame_id``.

    Each local inertia is rotated into the assembly frame, then shifted from
    its component center to the combined center using the canonical
    ``model_generation`` parallel-axis primitive.
    """
    normalized = _require_components(components)
    frame_id = require_identifier(assembly_frame_id, "assembly_frame_id")
    _require_common_frame(normalized, frame_id)
    masses = np.asarray([item.mass_properties.mass_kg for item in normalized])
    centers = np.asarray(
        [
            item.transform_to_club.transform_point(
                item.mass_properties.center_of_mass_m
            )
            for item in normalized
        ]
    )
    total_mass = float(np.sum(masses))
    center = np.sum(masses[:, None] * centers, axis=0) / total_mass
    inertia = _combined_inertia(normalized, masses, centers, center)
    return AssembledMassProperties(
        frame_id=frame_id,
        component_ids=tuple(item.component_id for item in normalized),
        total_mass_kg=total_mass,
        center_of_mass_m=tuple(center),
        inertia_at_com_kg_m2=inertia,
    )


@dataclass(frozen=True)
class ClubAssembly:
    """Versioned serializable club assembly with a declared length record."""

    assembly_id: str
    frame_id: str
    components: tuple[ClubComponent, ...]
    club_length: ClubLengthMeasurement

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "assembly_id", require_identifier(self.assembly_id, "assembly_id")
        )
        frame_id = require_identifier(self.frame_id, "frame_id")
        object.__setattr__(self, "frame_id", frame_id)
        components = _require_components(self.components)
        _require_common_frame(components, frame_id)
        object.__setattr__(self, "components", components)
        if not isinstance(self.club_length, ClubLengthMeasurement):
            raise TypeError("club_length must be ClubLengthMeasurement")

    @property
    def mass_properties(self) -> AssembledMassProperties:
        """Return mass properties assembled in :attr:`frame_id`."""
        return assemble_mass_properties(self.components, self.frame_id)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the current versioned JSON-compatible representation."""
        from .serialization import assembly_to_json_dict

        payload: dict[str, Any] = assembly_to_json_dict(self)
        return payload

    def to_json(self) -> str:
        """Return deterministic compact JSON for persistence or hashing."""
        from .serialization import assembly_to_json

        payload: str = assembly_to_json(self)
        return payload

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> ClubAssembly:
        """Load and validate current or supported legacy assembly data."""
        from .serialization import assembly_from_json_dict

        assembly: ClubAssembly = assembly_from_json_dict(data)
        return assembly

    @classmethod
    def from_json(cls, text: str) -> ClubAssembly:
        """Parse, migrate, and validate a JSON assembly document."""
        from .serialization import assembly_from_json

        assembly: ClubAssembly = assembly_from_json(text)
        return assembly


def _require_components(components: object) -> tuple[ClubComponent, ...]:
    """Normalize and validate a nonempty, unique component sequence."""
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise TypeError("components must be a sequence of ClubComponent values")
    normalized = tuple(components)
    if not normalized:
        raise ValueError("components must contain at least one component")
    if not all(isinstance(item, ClubComponent) for item in normalized):
        raise TypeError("components must contain only ClubComponent values")
    identifiers = tuple(item.component_id for item in normalized)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("component IDs must be unique")
    return normalized


def _require_common_frame(
    components: tuple[ClubComponent, ...], assembly_frame_id: str
) -> None:
    """Require every component transform to target the declared assembly frame."""
    if any(
        item.transform_to_club.to_frame_id != assembly_frame_id for item in components
    ):
        raise ValueError("every transform to_frame_id must equal assembly_frame_id")


def _combined_inertia(
    components: tuple[ClubComponent, ...],
    masses: np.ndarray,
    centers: np.ndarray,
    combined_center: np.ndarray,
) -> Matrix3:
    """Rotate, parallel-axis shift, and sum all component tensors."""
    total = np.zeros((3, 3))
    for item, mass, center in zip(components, masses, centers, strict=True):
        rotated = item.transform_to_club.rotate_inertia_tensor(
            item.mass_properties.inertia_at_com_kg_m2
        )
        shifted = parallel_axis(
            _tensor_to_dict(rotated),
            float(mass),
            tuple(float(value) for value in center - combined_center),
        )
        total += _dict_to_tensor(shifted)
    return require_inertia(total)


def _tensor_to_dict(tensor: Matrix3) -> dict[str, float]:
    """Adapt an immutable tensor to the canonical primitive's representation."""
    return {
        "ixx": tensor[0][0],
        "iyy": tensor[1][1],
        "izz": tensor[2][2],
        "ixy": tensor[0][1],
        "ixz": tensor[0][2],
        "iyz": tensor[1][2],
    }


def _dict_to_tensor(values: Mapping[str, float]) -> np.ndarray:
    """Adapt canonical inertia components back to a symmetric matrix."""
    if set(values) != set(_INERTIA_KEYS):
        raise AssertionError("parallel_axis returned an incomplete inertia tensor")
    return cast(
        np.ndarray,
        np.array(
            [
                [values["ixx"], values["ixy"], values["ixz"]],
                [values["ixy"], values["iyy"], values["iyz"]],
                [values["ixz"], values["iyz"], values["izz"]],
            ],
            dtype=float,
        ),
    )


__all__ = ["ClubAssembly", "assemble_mass_properties"]
