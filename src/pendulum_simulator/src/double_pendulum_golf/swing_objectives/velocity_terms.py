"""Named split of the velocity-product term into centrifugal and Coriolis parts.

``physics.coriolis_vector`` returns ``C(q, q̇) q̇`` as a single vector, so nothing
downstream can target one mechanism without the other. This module partitions
that same quantity into the two physically distinct drives of a golf downswing,
without changing the physics:

* **Centrifugal** — squared-velocity terms. The wrist entry
  ``+mu*sin(phi)*dtheta1**2`` is the passive release: arm rotation flinging the
  club open, growing with the square of arm speed and needing no muscular effort.
* **Coriolis** — the velocity cross term ``2*h*dtheta1*dphi``, present only in the
  hub row. It is the kinetic chain: as the club uncocks it drains angular
  momentum out of the arms.

With ``h = -mu*sin(phi)`` and ``mu = (m2 + mClub)*L1*L2`` matching
``physics.mass_matrix``, the shipped combined vector is

.. code-block:: text

    c1 = h * (2*dtheta1*dphi + dphi**2)
    c2 = -h * dtheta1**2

so the partition is ``[h*dphi**2, -h*dtheta1**2]`` plus ``[2*h*dtheta1*dphi, 0]``.
The sum is required to reproduce ``coriolis_vector`` exactly; that closure is the
contract preventing this module from drifting away from the physics kernel it
describes, and it is enforced both here and in ``tests/test_swing_velocity_terms.py``.

Sign convention: all vectors are expressed on the **left-hand side** of
``M(q) q̈ + C(q, q̇) q̇ + G(q) = tau``, matching ``physics.coriolis_vector``.

Closes #4767.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from double_pendulum_golf.physics import PendulumParams, coriolis_vector

__all__ = [
    "VelocityTerms",
    "coupling_constant",
    "centrifugal_vector",
    "coriolis_only_vector",
    "decompose_velocity_terms",
]

FloatArray = npt.NDArray[np.float64]

#: Tolerance for the partition-closure postcondition, in N·m.
_CLOSURE_ATOL = 1e-9


@dataclass(frozen=True, slots=True)
class VelocityTerms:
    """Partition of ``C(q, q̇) q̇`` into its two named mechanisms.

    Attributes:
        centrifugal: Squared-velocity contribution ``[hub, wrist]`` in N·m.
        coriolis: Velocity cross-product contribution ``[hub, wrist]`` in N·m.
            The wrist entry is always exactly zero.

    Invariant:
        ``centrifugal + coriolis`` equals ``physics.coriolis_vector`` for the
        same arguments.
    """

    centrifugal: FloatArray
    coriolis: FloatArray

    @property
    def total(self) -> FloatArray:
        """Combined velocity-product vector ``C(q, q̇) q̇`` in N·m."""
        return self.centrifugal + self.coriolis


def _require_finite(phi: float, dtheta1: float, dphi: float) -> None:
    """Reject non-finite kinematic inputs.

    Pre: none.
    Post: raises ``ValueError`` unless all three arguments are finite.
    """
    if not all(np.isfinite(value) for value in (phi, dtheta1, dphi)):
        raise ValueError(f"phi, dtheta1 and dphi must be finite, got {phi}, {dtheta1}, {dphi}")


def coupling_constant(params: PendulumParams) -> float:
    """Return the inertial coupling constant ``mu = (m2 + mClub) * L1 * L2``.

    This single constant scales every centrifugal and Coriolis term in the model,
    so it is the quantitative measure of how strongly the arms and club are
    dynamically coupled. It matches the off-diagonal coupling built into
    ``physics.mass_matrix``, which treats the clubhead as a point mass at the tip.

    Args:
        params: Double pendulum physical parameters.

    Returns:
        Coupling constant in kg·m².

    Post: strictly positive, because masses and lengths are positive by contract.
    """
    effective_distal_mass = params.m2 + params.mClub
    return float(effective_distal_mass * params.L1 * params.L2)


def _coupling_sine(phi: float, params: PendulumParams) -> float:
    """Return ``h = -mu * sin(phi)``, the shared factor of both mechanisms."""
    return -coupling_constant(params) * float(np.sin(phi))


def centrifugal_vector(
    phi: float, dtheta1: float, dphi: float, params: PendulumParams
) -> FloatArray:
    """Compute the centrifugal (squared-velocity) part of ``C(q, q̇) q̇``.

    The wrist entry is the passive release drive: it depends on ``dtheta1**2`` and
    is completely independent of how fast the wrists are already uncocking.

    Args:
        phi: Wrist cock angle in rad, club relative to arms.
        dtheta1: Arm angular velocity in rad/s.
        dphi: Wrist uncocking rate in rad/s.
        params: Double pendulum physical parameters.

    Returns:
        Length-2 array ``[hub, wrist]`` in N·m, on the left-hand side.

    Pre: ``phi``, ``dtheta1`` and ``dphi`` are finite.
    Post: both entries are finite.
    """
    _require_finite(phi, dtheta1, dphi)
    coupling_sine = _coupling_sine(phi, params)
    result = np.array([coupling_sine * dphi**2, -coupling_sine * dtheta1**2], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Centrifugal vector has non-finite values: {result}")
    return result


def coriolis_only_vector(
    phi: float, dtheta1: float, dphi: float, params: PendulumParams
) -> FloatArray:
    """Compute the true Coriolis (velocity cross-product) part of ``C(q, q̇) q̇``.

    Only the hub row carries a Coriolis term. Because it is proportional to
    ``dtheta1 * dphi``, it can only act while the club is actually uncocking, and
    it reverses sign if either rate reverses.

    Args:
        phi: Wrist cock angle in rad, club relative to arms.
        dtheta1: Arm angular velocity in rad/s.
        dphi: Wrist uncocking rate in rad/s.
        params: Double pendulum physical parameters.

    Returns:
        Length-2 array ``[hub, 0.0]`` in N·m, on the left-hand side.

    Pre: ``phi``, ``dtheta1`` and ``dphi`` are finite.
    Post: entries are finite and the wrist entry is exactly zero.
    """
    _require_finite(phi, dtheta1, dphi)
    hub_term = 2.0 * _coupling_sine(phi, params) * dtheta1 * dphi
    if not np.isfinite(hub_term):
        raise ValueError(f"Coriolis hub term is non-finite: {hub_term}")
    return np.array([hub_term, 0.0], dtype=np.float64)


def decompose_velocity_terms(
    phi: float, dtheta1: float, dphi: float, params: PendulumParams
) -> VelocityTerms:
    """Partition ``C(q, q̇) q̇`` into its centrifugal and Coriolis mechanisms.

    Args:
        phi: Wrist cock angle in rad, club relative to arms.
        dtheta1: Arm angular velocity in rad/s.
        dphi: Wrist uncocking rate in rad/s.
        params: Double pendulum physical parameters.

    Returns:
        The immutable two-part decomposition.

    Pre: ``phi``, ``dtheta1`` and ``dphi`` are finite.
    Post: ``total`` reproduces ``physics.coriolis_vector`` to ``1e-9`` N·m.
    """
    terms = VelocityTerms(
        centrifugal=centrifugal_vector(phi, dtheta1, dphi, params),
        coriolis=coriolis_only_vector(phi, dtheta1, dphi, params),
    )
    _ensure_partition_closes(terms, phi, dtheta1, dphi, params)
    return terms


def _ensure_partition_closes(
    terms: VelocityTerms,
    phi: float,
    dtheta1: float,
    dphi: float,
    params: PendulumParams,
) -> None:
    """Verify the split still reproduces the shipped combined vector.

    Guards against the physics kernel and this decomposition diverging — for
    instance if the native Rust backend changed its convention.
    """
    combined = coriolis_vector(phi, dtheta1, dphi, params)
    if not np.allclose(terms.total, combined, atol=_CLOSURE_ATOL):
        raise ValueError(
            "Velocity-term partition does not close against physics.coriolis_vector: "
            f"{terms.total} vs {combined}"
        )
