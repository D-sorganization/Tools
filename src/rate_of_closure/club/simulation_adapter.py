"""Strict ClubAssembly binding adapter for the impact-solver boundary."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from rate_of_closure._contracts import ensure, require

from .assembly_binding import SELECTED_HEAD_FRAME_ID, ClubAssemblyBinding
from .types import ClubSpec

APP_FRAME_ID = "rate_of_closure.app"
CapabilityStatus = Literal["available", "unavailable", "not_used"]
FloatArray: TypeAlias = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SimulationCapabilityUse:
    """Availability and consumption decision for one solver property."""

    status: CapabilityStatus
    consumed: bool
    reason: str

    def __post_init__(self) -> None:
        require(
            self.status in ("available", "unavailable", "not_used"),
            "unsupported simulation capability status",
            self.status,
        )
        require(isinstance(self.consumed, bool), "consumed must be bool")
        require(
            isinstance(self.reason, str) and bool(self.reason.strip()),
            "capability reason must be nonempty",
        )
        require(
            self.consumed == (self.status == "available"),
            "only available properties can be consumed",
        )


@dataclass(frozen=True)
class WorldFromHeadAttitude:
    """Explicit complete rotation from the selected head to app frame."""

    from_frame_id: str
    to_frame_id: str
    rotation: FloatArray
    provenance: str

    def __post_init__(self) -> None:
        require(
            self.from_frame_id == SELECTED_HEAD_FRAME_ID,
            f"attitude must transform from {SELECTED_HEAD_FRAME_ID}",
            self.from_frame_id,
        )
        require(
            self.to_frame_id == APP_FRAME_ID,
            f"attitude must transform to {APP_FRAME_ID}",
            self.to_frame_id,
        )
        require(
            isinstance(self.provenance, str) and bool(self.provenance.strip()),
            "attitude provenance must be nonempty",
        )
        rotation = np.array(self.rotation, dtype=np.float64, copy=True)
        require(rotation.shape == (3, 3), "attitude rotation must be 3x3")
        require(bool(np.all(np.isfinite(rotation))), "attitude rotation must be finite")
        require(
            bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)),
            "attitude rotation must be orthonormal",
        )
        require(
            bool(np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)),
            "attitude rotation must be proper",
        )
        rotation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)


@dataclass(frozen=True)
class ClubAssemblyImpactInputs:
    """Validated values and explicit non-consumption decisions for impact."""

    head_mass_kg: float
    head_inertia_tensor_app_kg_m2: FloatArray | None
    head_inertia: SimulationCapabilityUse
    head_center_of_mass: SimulationCapabilityUse
    assembly_mass_properties: SimulationCapabilityUse

    def to_json_dict(self) -> dict[str, object]:
        """Return the capability ledger without duplicating binding payloads."""

        def capability(value: SimulationCapabilityUse) -> dict[str, object]:
            return {
                "status": value.status,
                "consumed": value.consumed,
                "reason": value.reason,
            }

        return {
            "head_mass_kg": self.head_mass_kg,
            "head_inertia": capability(self.head_inertia),
            "head_center_of_mass": capability(self.head_center_of_mass),
            "assembly_mass_properties": capability(self.assembly_mass_properties),
        }

    def without_impact(self) -> ClubAssemblyImpactInputs:
        """Return the same authority record with all solver use disabled."""
        reason = "not consumed because no club-ball impact occurred"
        return replace(
            self,
            head_inertia_tensor_app_kg_m2=None,
            head_inertia=SimulationCapabilityUse("not_used", False, reason),
            head_center_of_mass=SimulationCapabilityUse("not_used", False, reason),
            assembly_mass_properties=SimulationCapabilityUse("not_used", False, reason),
        )


def _unavailable(reason: str) -> SimulationCapabilityUse:
    return SimulationCapabilityUse("unavailable", False, reason)


def adapt_club_assembly_for_impact(
    spec: ClubSpec,
    binding: ClubAssemblyBinding | None,
    attitude: WorldFromHeadAttitude | None,
) -> ClubAssemblyImpactInputs:
    """Adapt only solver-supported head properties; never use assembly inertia."""
    require(isinstance(spec, ClubSpec), "spec must be a ClubSpec")
    if binding is None:
        require(attitude is None, "head attitude requires an assembly binding")
        missing = "unavailable because no validated ClubAssembly binding is configured"
        return ClubAssemblyImpactInputs(
            head_mass_kg=spec.head_mass_kg,
            head_inertia_tensor_app_kg_m2=None,
            head_inertia=_unavailable(missing),
            head_center_of_mass=_unavailable(missing),
            assembly_mass_properties=_unavailable(missing),
        )
    require(
        isinstance(binding, ClubAssemblyBinding),
        "binding must be a ClubAssemblyBinding or None",
    )
    binding.assert_matches(spec)
    head = binding.head_properties_in_selected_frame()
    tensor = _head_tensor_in_app_frame(head.inertia_at_com_kg_m2, attitude)
    tensor_use = (
        SimulationCapabilityUse(
            "available",
            True,
            "validated head-CG tensor rotated into rate_of_closure.app",
        )
        if tensor is not None
        else _unavailable(
            "swing source does not declare a complete "
            "world-from-rate_of_closure.head attitude"
        )
    )
    return ClubAssemblyImpactInputs(
        head_mass_kg=head.mass_kg,
        head_inertia_tensor_app_kg_m2=tensor,
        head_inertia=tensor_use,
        head_center_of_mass=_unavailable(
            "impact solver does not accept a full head-CG vector with its "
            "declared datum"
        ),
        assembly_mass_properties=_unavailable(
            "impact solver requires head properties; must not substitute "
            "assembled-club mass, CG, or inertia"
        ),
    )


def _head_tensor_in_app_frame(
    tensor_head_kg_m2: object,
    attitude: WorldFromHeadAttitude | None,
) -> FloatArray | None:
    if attitude is None:
        return None
    tensor = np.asarray(tensor_head_kg_m2, dtype=np.float64)
    rotation = attitude.rotation
    transformed: FloatArray = rotation @ tensor @ rotation.T
    ensure(
        bool(np.allclose(transformed, transformed.T, atol=1e-12)),
        "transformed head tensor must remain symmetric",
    )
    transformed.setflags(write=False)
    return transformed


__all__ = [
    "APP_FRAME_ID",
    "ClubAssemblyImpactInputs",
    "SimulationCapabilityUse",
    "WorldFromHeadAttitude",
    "adapt_club_assembly_for_impact",
]
