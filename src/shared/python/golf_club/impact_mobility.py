"""Detached rigid-body contact reference for IA-T1 (#5069).

All vectors and COM inertia tensors must use the same orthonormal frame.
Contact inverse mass maps an impulse [N s] to point velocity change [m/s].
These instantaneous, frictionless references contain no shaft, face compliance,
prestress, grip constraint or sound radiation; they do not bound those systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ._validation import (
    Matrix3,
    Vector3,
    require_finite_float,
    require_inertia,
    require_vector3,
)

__all__ = [
    "RigidContactBody",
    "contact_inverse_mass",
    "normal_effective_mass",
    "normal_impulse",
]

_UNIT_NORMAL_TOLERANCE = 1e-10


@dataclass(frozen=True)
class RigidContactBody:
    """Immutable mass and geometry at one contact point.

    Preconditions: positive mass [kg], symmetric physically realizable positive
    definite COM inertia [kg m²], finite COM-to-contact offset [m], common frame.
    Postconditions: caller-owned arrays are copied to immutable tuples.
    Degenerate line/point inertias are refused; no pseudoinverse is inferred.
    """

    mass_kg: float
    inertia_at_com_kg_m2: Matrix3
    contact_offset_m: Vector3

    def __post_init__(self) -> None:
        mass = require_finite_float(self.mass_kg, "mass_kg", positive=True)
        inertia = require_inertia(self.inertia_at_com_kg_m2)
        offset = require_vector3(self.contact_offset_m, "contact_offset_m")
        try:
            np.linalg.cholesky(np.asarray(inertia))
        except np.linalg.LinAlgError as error:
            raise ValueError("COM inertia must be positive definite") from error
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "inertia_at_com_kg_m2", inertia)
        object.__setattr__(self, "contact_offset_m", offset)


def contact_inverse_mass(body: RigidContactBody) -> NDArray[np.float64]:
    """Return W = I/m - [r]× I_COM⁻¹ [r]× in inverse kilograms.

    Preconditions: a validated detached RigidContactBody.
    Postconditions: a fresh finite symmetric positive definite 3-by-3 matrix.
    The cross-product construction retains products of inertia and coupling.
    """
    if not isinstance(body, RigidContactBody):
        raise TypeError("body must be RigidContactBody")
    offset_x, offset_y, offset_z = body.contact_offset_m
    cross = np.array(
        [[0, -offset_z, offset_y], [offset_z, 0, -offset_x], [-offset_y, offset_x, 0]],
        dtype=float,
    )
    inertia = np.asarray(body.inertia_at_com_kg_m2)
    mobility = np.eye(3) / body.mass_kg - cross @ np.linalg.solve(inertia, cross)
    if not np.all(np.isfinite(mobility)):
        raise ValueError("contact inverse mass exceeds numerical range")
    return np.asarray(0.5 * (mobility + mobility.T), dtype=np.float64)


def normal_effective_mass(body: RigidContactBody, normal: Vector3) -> float:
    """Return 1/(nᵀ W n) [kg] for a unit contact normal in the body's frame.

    Preconditions: body as above; finite unit normal (never silently normalized).
    Postconditions: finite positive directional mass. This is not nᵀ W⁻¹ n:
    tangential velocity is free to change under a normal impulse.
    """
    direction = np.asarray(require_vector3(normal, "normal"))
    if not np.isclose(
        np.linalg.norm(direction), 1.0, rtol=0, atol=_UNIT_NORMAL_TOLERANCE
    ):
        raise ValueError("normal must be a unit vector")
    inverse_mass = float(direction @ contact_inverse_mass(body) @ direction)
    return require_finite_float(1.0 / inverse_mass, "effective_mass", positive=True)


def normal_impulse(
    closing_speed_mps: float,
    first_effective_mass_kg: float,
    second_effective_mass_kg: float,
    restitution: float,
) -> float:
    """Return compressive impulse [N s] for a frictionless two-body collision.

    Preconditions: finite closing speed (positive for approach), positive
    directional masses in the shared normal, restitution in [0, 1]. No other
    impulses or changing configuration during the instantaneous collision.
    Postconditions: nonnegative impulse; separating/touching states return zero.
    Positive approach obeys Newton restitution and dissipates
    0.5*reduced_mass*(1-restitution²)*closing_speed² of kinetic energy.
    """
    speed = require_finite_float(closing_speed_mps, "closing_speed_mps")
    first = require_finite_float(first_effective_mass_kg, "first_mass", positive=True)
    second = require_finite_float(
        second_effective_mass_kg, "second_mass", positive=True
    )
    cor = require_finite_float(restitution, "restitution")
    if not 0 <= cor <= 1:
        raise ValueError("restitution must be in [0, 1]")
    # Ratio form avoids the overflowing mass product in m1*m2/(m1+m2).
    smaller, larger = min(first, second), max(first, second)
    reduced_mass = smaller / (1 + smaller / larger)
    result = (1 + cor) * max(0.0, speed) * reduced_mass
    return require_finite_float(result, "impulse")
