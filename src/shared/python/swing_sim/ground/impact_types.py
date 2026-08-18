"""Typed records for one passive sphere-plane impulse."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import StrEnum
else:
    from shared.python.compatibility import StrEnum

from .contract_types import GroundContactState, Vector3

_ENERGY_ABSOLUTE_TOLERANCE_J = 1e-10
_ENERGY_RELATIVE_TOLERANCE = 1e-10


def _energy_balance_tolerance_j(*values: float) -> float:
    return _ENERGY_ABSOLUTE_TOLERANCE_J + _ENERGY_RELATIVE_TOLERANCE * max(
        (abs(value) for value in values),
        default=0.0,
    )


class ImpactRegime(StrEnum):
    """Tangential contact regime selected by the Coulomb law."""

    STICKING = "sticking"
    SLIDING = "sliding"


class ImpactRejectionReason(StrEnum):
    """Typed reason that an input state is not an admissible impact."""

    GRAZING = "grazing"
    OUTGOING = "outgoing"


class ImpactStateError(ValueError):
    """Reject a grazing or outgoing state before applying an impulse."""

    def __init__(self, reason: ImpactRejectionReason) -> None:
        self.reason = reason
        super().__init__(f"impact state is {reason.value}")


@dataclass(frozen=True)
class SphereProperties:
    """Rigid-sphere mass, radius, and dimensionless inertia factor."""

    radius_m: float
    mass_kg: float
    rotational_inertia_factor: float

    def __post_init__(self) -> None:
        for name in ("radius_m", "mass_kg", "rotational_inertia_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and positive")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if self.rotational_inertia_factor > 1.0:
            raise ValueError("rotational_inertia_factor must lie within (0, 1]")

    @property
    def inertia_kg_m2(self) -> float:
        """Return the isotropic rigid-sphere moment of inertia."""
        return self.rotational_inertia_factor * self.mass_kg * self.radius_m**2

    @property
    def tangential_effective_mass_kg(self) -> float:
        """Return the sphere contact effective mass in either tangent axis."""
        factor = self.rotational_inertia_factor
        return self.mass_kg * factor / (factor + 1.0)


@dataclass(frozen=True)
class ImpactEnergyLedger:
    """Kinetic-energy and moving-boundary accounting in joules."""

    kinetic_before_j: float
    kinetic_after_j: float
    boundary_work_j: float
    dissipation_j: float

    def __post_init__(self) -> None:
        values = (
            self.kinetic_before_j,
            self.kinetic_after_j,
            self.boundary_work_j,
            self.dissipation_j,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("impact energy ledger must be finite")
        if self.kinetic_before_j < 0.0 or self.kinetic_after_j < 0.0:
            raise ValueError("impact kinetic energy must be nonnegative")
        if self.dissipation_j < 0.0:
            raise ValueError("impact dissipation must be nonnegative")
        expected_dissipation = (
            self.kinetic_before_j + self.boundary_work_j - self.kinetic_after_j
        )
        tolerance = _energy_balance_tolerance_j(
            self.kinetic_before_j,
            self.kinetic_after_j,
            self.boundary_work_j,
            self.dissipation_j,
        )
        if abs(self.dissipation_j - expected_dissipation) > tolerance:
            raise ValueError("impact dissipation must match the energy balance")


@dataclass(frozen=True)
class ImpactImpulseResult:
    """Validated before/after state and impulse diagnostics."""

    state_before: GroundContactState
    state_after: GroundContactState
    regime: ImpactRegime
    normal_impulse_n_s: float
    tangential_impulse_n_s: Vector3
    total_impulse_n_s: Vector3
    contact_velocity_before_m_s: Vector3
    contact_velocity_after_m_s: Vector3
    effective_restitution: float
    friction_utilization: float
    energy: ImpactEnergyLedger

    def __post_init__(self) -> None:
        if type(self.state_before) is not GroundContactState:
            raise ValueError("impact before-state must be an exact contact state")
        if type(self.state_after) is not GroundContactState:
            raise ValueError("impact after-state must be an exact contact state")
        if type(self.energy) is not ImpactEnergyLedger:
            raise ValueError("impact result requires an exact energy ledger")
        object.__setattr__(self, "regime", ImpactRegime(self.regime))
        vectors = (
            self.tangential_impulse_n_s,
            self.total_impulse_n_s,
            self.contact_velocity_before_m_s,
            self.contact_velocity_after_m_s,
        )
        if any(len(vector) != 3 for vector in vectors):
            raise ValueError("impact vectors must contain three components")
        if not all(math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("impact vectors must be finite")
        if self.state_before.frame is not self.state_after.frame:
            raise ValueError("impact state frames must match")
        if self.state_before.time_s != self.state_after.time_s:
            raise ValueError("an impulse cannot advance time")
        if not math.isfinite(self.normal_impulse_n_s) or self.normal_impulse_n_s <= 0.0:
            raise ValueError("normal impulse must be positive")
        if not math.isfinite(self.effective_restitution):
            raise ValueError("effective restitution must be finite")
        if not 0.0 <= self.effective_restitution <= 1.0:
            raise ValueError("effective restitution must lie within [0, 1]")
        if not math.isfinite(self.friction_utilization):
            raise ValueError("friction utilization must be finite")
        if not 0.0 <= self.friction_utilization <= 1.0 + 1e-10:
            raise ValueError("friction utilization must lie within [0, 1]")


__all__ = [
    "ImpactEnergyLedger",
    "ImpactImpulseResult",
    "ImpactRegime",
    "ImpactRejectionReason",
    "ImpactStateError",
    "SphereProperties",
]
