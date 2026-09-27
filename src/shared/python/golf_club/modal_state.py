"""Modal basis, shaft modal state and M-orthogonal projection (Tools #5353).

The represented energy is the declared quadratic form
``0.5 qd^T M_r qd + 0.5 q^T K_r q``.
The M-orthogonal projection preserves in-span energy and explicitly accounts for
out-of-span residuals without discarding or concealing them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._pre_impact_contracts import (
    FloatArray,
    Quantity,
    fail,
    finite_array,
    identifier,
    parse_quantity,
    positive_definite,
    positive_semidefinite,
    reject_unknown,
    strictly_increasing,
)

_NORMALIZATIONS = frozenset({"mass", "none"})
_MASS_NORMALIZATION_TOLERANCE = 1e-9


@dataclass(frozen=True, eq=False)
class ModalBasis:
    """Identity and reduced quadratic forms of a shaft mode basis."""

    basis_id: str
    version: str
    normalization: str
    dimension: int
    generalized_mass: FloatArray
    generalized_stiffness: FloatArray

    def __post_init__(self) -> None:
        identifier(self.basis_id, "basis_id")
        identifier(self.version, "basis version")
        if self.normalization not in _NORMALIZATIONS:
            fail("normalization", f"normalization must be in {sorted(_NORMALIZATIONS)}")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            fail("type", "dimension must be an integer")
        if self.dimension < 1:
            fail("out_of_range", "dimension must be >= 1")
        size = self.dimension
        mass = positive_definite(self.generalized_mass, size, "generalized_mass")
        stiffness = positive_semidefinite(
            self.generalized_stiffness, size, "generalized_stiffness"
        )
        if self.normalization == "mass" and not np.allclose(
            mass, np.eye(size), rtol=0.0, atol=_MASS_NORMALIZATION_TOLERANCE
        ):
            fail("normalization", "mass-normalized basis needs identity modal mass")
        object.__setattr__(self, "generalized_mass", mass)
        object.__setattr__(self, "generalized_stiffness", stiffness)

    def energy_j(self, amplitudes: FloatArray, velocities: FloatArray) -> float:
        """Represented energy ``0.5 qd^T M qd + 0.5 q^T K q`` in joules."""
        kinetic, potential = self._energies(amplitudes, velocities)
        return kinetic + potential

    def _energies(
        self, amplitudes: FloatArray, velocities: FloatArray
    ) -> tuple[float, float]:
        q = finite_array(amplitudes, (self.dimension,), "amplitudes")
        qd = finite_array(velocities, (self.dimension,), "velocities")
        kinetic = 0.5 * float(qd @ self.generalized_mass @ qd)
        return kinetic, 0.5 * float(q @ self.generalized_stiffness @ q)

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis_id": self.basis_id,
            "version": self.version,
            "normalization": self.normalization,
            "dimension": self.dimension,
            "generalized_mass": self.generalized_mass.tolist(),
            "generalized_stiffness": self.generalized_stiffness.tolist(),
        }


@dataclass(frozen=True, eq=False)
class ShaftState:
    """Reduced modal state and prescribed axial-force field of the shaft."""

    basis: ModalBasis
    basis_id: str
    basis_version: str
    amplitudes: Quantity
    velocities: Quantity
    axial_stations_m: FloatArray
    axial_force_n: Quantity

    def __post_init__(self) -> None:
        if (self.basis_id, self.basis_version) != (
            self.basis.basis_id,
            self.basis.version,
        ):
            fail("basis_mismatch", "modal state basis id/version must match basis")
        for name in ("amplitudes", "velocities"):
            quantity = getattr(self, name)
            if not quantity.is_absent and quantity.value.shape != (
                self.basis.dimension,
            ):
                fail("basis_mismatch", f"{name} must match basis dimension")
        stations = self.axial_stations_m
        if stations.ndim != 1 or stations.size < 1 or float(stations[0]) < 0.0:
            fail("shape", "axial_stations_m must be nonnegative stations")
        strictly_increasing(stations, "axial_stations_m")
        axial_force = self.axial_force_n
        if not axial_force.is_absent and axial_force.value.shape != stations.shape:
            fail("shape", "axial_force_n must have one value per station")

    def modal_energy_j(self) -> float:
        """Represented modal energy; absent elastic state raises, never zero."""
        return self.basis.energy_j(self.amplitudes.value, self.velocities.value)


@dataclass(frozen=True, eq=False)
class ModalProjection:
    """Mass-orthogonal projection result with residuals and energy ledger."""

    amplitudes: FloatArray
    velocities: FloatArray
    displacement_residual: float
    velocity_residual: float
    full_kinetic_energy_j: float
    full_potential_energy_j: float
    represented_kinetic_energy_j: float
    represented_potential_energy_j: float


def _relative_residual(
    residual: FloatArray, full: FloatArray, mass: FloatArray
) -> float:
    full_norm = float(np.sqrt(full @ mass @ full))
    residual_norm = float(np.sqrt(max(float(residual @ mass @ residual), 0.0)))
    return residual_norm / full_norm if full_norm > 0.0 else residual_norm


def project_onto_basis(
    basis: ModalBasis,
    shapes: object,
    full_mass: object,
    full_stiffness: object,
    displacement: object,
    velocity: object,
) -> ModalProjection:
    """Project a full-coordinate state onto ``basis`` without resetting energy.

    Preconditions: ``shapes`` (n, dim) reproduce the basis quadratic forms,
    ``Phi^T M Phi == M_r`` and ``Phi^T K Phi == K_r``; ``M`` is SPD, ``K`` PSD.
    Postconditions: ``q = (Phi^T M Phi)^-1 Phi^T M u`` (M-orthogonal), relative
    M-norm residuals, and full versus represented energies. An in-span state has
    zero residual and identical energy; out-of-span energy is reported, not
    silently discarded.
    """
    phi = np.asarray(shapes, dtype=float)
    if phi.ndim != 2 or phi.shape[1] != basis.dimension:
        fail("basis_mismatch", "shapes must have one column per basis mode")
    size = phi.shape[0]
    phi = finite_array(phi, (size, basis.dimension), "shapes")
    mass = positive_definite(full_mass, size, "full_mass")
    stiffness = positive_semidefinite(full_stiffness, size, "full_stiffness")
    for reduced, expected, name in (
        (phi.T @ mass @ phi, basis.generalized_mass, "mass"),
        (phi.T @ stiffness @ phi, basis.generalized_stiffness, "stiffness"),
    ):
        scale = max(float(np.max(np.abs(expected))), 1.0)
        if not np.allclose(reduced, expected, rtol=0.0, atol=1e-9 * scale):
            fail("basis_mismatch", f"shapes do not reproduce the basis {name}")
    u = finite_array(displacement, (size,), "displacement")
    ud = finite_array(velocity, (size,), "velocity")
    projector = np.linalg.solve(basis.generalized_mass, phi.T @ mass)
    q, qd = projector @ u, projector @ ud
    kinetic, potential = basis._energies(q, qd)
    for array in (q, qd):
        array.setflags(write=False)
    return ModalProjection(
        amplitudes=q,
        velocities=qd,
        displacement_residual=_relative_residual(u - phi @ q, u, mass),
        velocity_residual=_relative_residual(ud - phi @ qd, ud, mass),
        full_kinetic_energy_j=0.5 * float(ud @ mass @ ud),
        full_potential_energy_j=0.5 * float(u @ stiffness @ u),
        represented_kinetic_energy_j=kinetic,
        represented_potential_energy_j=potential,
    )


def _basis_from_dict(raw: object) -> ModalBasis:
    fields = {"basis_id", "version", "normalization", "dimension"}
    data = reject_unknown(
        raw, frozenset(fields | {"generalized_mass", "generalized_stiffness"}), "basis"
    )
    size = data["dimension"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        fail("type", "basis dimension must be a positive integer")
    return ModalBasis(
        basis_id=data["basis_id"],
        version=data["version"],
        normalization=data["normalization"],
        dimension=size,
        generalized_mass=finite_array(
            data["generalized_mass"], (size, size), "generalized_mass"
        ),
        generalized_stiffness=finite_array(
            data["generalized_stiffness"], (size, size), "generalized_stiffness"
        ),
    )


def _vector_length(raw: object) -> int:
    value = raw.get("value") if isinstance(raw, Mapping) else None
    return len(value) if isinstance(value, (list, tuple, np.ndarray)) else -1


def _shaft_from_dict(raw: object) -> ShaftState:
    fields = {"basis", "basis_id", "basis_version", "amplitudes", "velocities"}
    data = reject_unknown(
        raw, frozenset(fields | {"axial_stations_m", "axial_force_n"}), "shaft"
    )
    stations = data["axial_stations_m"]
    count = len(stations) if isinstance(stations, (list, tuple, np.ndarray)) else -1
    return ShaftState(
        basis=_basis_from_dict(data["basis"]),
        basis_id=identifier(data["basis_id"], "basis_id"),
        basis_version=identifier(data["basis_version"], "basis_version"),
        amplitudes=parse_quantity(
            data["amplitudes"],
            "shaft.amplitudes",
            (_vector_length(data["amplitudes"]),),
        ),
        velocities=parse_quantity(
            data["velocities"],
            "shaft.velocities",
            (_vector_length(data["velocities"]),),
        ),
        axial_stations_m=finite_array(stations, (count,), "axial_stations_m"),
        axial_force_n=parse_quantity(
            data["axial_force_n"],
            "shaft.axial_force_n",
            (_vector_length(data["axial_force_n"]),),
        ),
    )


__all__ = [
    "ModalBasis",
    "ModalProjection",
    "ShaftState",
    "project_onto_basis",
]
