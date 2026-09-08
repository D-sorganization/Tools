"""Small-strain section elasticity with a complete nodal SE(3) energy tangent.

This private element has no inertia, applied loads, equilibrium solver or
experimental validity claim. Its section stiffness must be supplied explicitly.
See docs/development/impact-acoustics/LOADED_STATE_REVIEW.md for derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import Matrix6, Vector6, finite_array, vector6
from ._shaft_se3 import (
    _rigid_pose,
    exp_twist,
    log_pose,
    right_jacobian,
    right_jacobian_derivative,
    twist_ad,
)
from ._validation import require_finite_float


def _section_stiffness(value: object) -> Matrix6:
    matrix = finite_array(value, (6, 6), "section stiffness")
    # Exact symmetry is an input contract; never silently repair a physical law.
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("section stiffness must be symmetric")
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("section stiffness must be positive definite") from error
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _strain_map(relative: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.linalg.solve(right_jacobian(-relative), np.eye(6))
    right = np.linalg.solve(right_jacobian(relative), np.eye(6))
    return np.hstack((-left, right)), left, right


def _map_derivative(
    relative: np.ndarray, direction: np.ndarray, inverses: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    left, right = inverses
    left_derivative = right_jacobian_derivative(-relative, -direction)
    right_derivative = right_jacobian_derivative(relative, direction)
    return np.hstack((left @ left_derivative @ left, -right @ right_derivative @ right))


def _chart_correction(gradient: np.ndarray, direction: np.ndarray) -> np.ndarray:
    # D J_r(0)[dq] = -ad(dq)/2 for each fixed nodal exponential chart.
    return np.asarray(
        -0.5
        * np.r_[
            twist_ad(direction[:6]).T @ gradient[:6],
            twist_ad(direction[6:]).T @ gradient[6:],
        ]
    )


@dataclass(frozen=True)
class SectionLinearization:
    """Fresh energy derivatives at q=0 for H_i(q)=H_i(0) Exp(q_i).

    Gradient is conjugate to nodal local translations [m] and rotations [rad].
    Tangent is the full fixed-chart energy Hessian, not a moving-basis force
    Jacobian. Material tangent is retained separately for geometric-term audits.
    """

    energy_j: float
    gradient: np.ndarray
    tangent: np.ndarray
    material_tangent: np.ndarray


@dataclass(frozen=True)
class SectionElement:
    """Uniform constant-strain elastic element in material section coordinates.

    Reference twist d0 is linear-first, in metres and radians, for length L>0.
    Strain is (Log(H_left^-1 H_right)-d0)/L: first three entries dimensionless,
    last three in 1/m. The symmetric positive-definite 6x6 stiffness maps this
    strain to section force [N] and moment [N m]; its blocks have units N, N m,
    and N m². Coupled and shear stiffness are explicit, never inferred from EI.
    The quadratic law assumes small material strains despite finite rotations.
    """

    length_m: float
    reference_twist: Vector6
    stiffness: Matrix6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "length_m",
            require_finite_float(self.length_m, "element length", positive=True),
        )
        reference = vector6(self.reference_twist, "reference twist")
        exp_twist(reference)  # Validate the reference logarithm's branch domain.
        object.__setattr__(self, "reference_twist", reference)
        object.__setattr__(self, "stiffness", _section_stiffness(self.stiffness))

    def _relative(self, left: object, right: object) -> np.ndarray:
        return log_pose(np.linalg.solve(_rigid_pose(left), _rigid_pose(right)))

    def _constitutive(self, relative: np.ndarray) -> tuple[float, np.ndarray]:
        strain = self._strain(relative)
        with np.errstate(over="ignore", invalid="ignore"):
            force = np.asarray(self.stiffness) @ strain
            energy = self.length_m * float(strain @ force) / 2
        return (
            require_finite_float(energy, "section energy"),
            finite_array(force, (6,), "section resultant"),
        )

    def _strain(self, relative: np.ndarray) -> np.ndarray:
        return finite_array(
            (relative - self.reference_twist) / self.length_m, (6,), "section strain"
        )

    def strain(self, left: object, right: object) -> np.ndarray:
        """Return fresh material strain: extension/shear, then curvature [1/m].

        Numerical log-chart admissibility does not establish physical validity;
        callers must compare these components with explicit material limits.
        """
        return self._strain(self._relative(left, right))

    def energy(self, left: object, right: object) -> float:
        """Return finite elastic energy [J] for proper poses in a common frame."""
        return self._constitutive(self._relative(left, right))[0]

    def linearize(self, left: object, right: object) -> SectionLinearization:
        """Return analytic virtual work and complete fixed-chart curvature.

        No finite differences, imposed symmetry, eigenvalue clipping or omitted
        prestress terms are used. This is an internal elastic tangent only;
        external-load and kinetic derivatives remain separate responsibilities.
        """
        relative = self._relative(left, right)
        energy, resultant = self._constitutive(relative)
        mapping, inverse_left, inverse_right = _strain_map(relative)
        gradient = mapping.T @ resultant
        material = mapping.T @ np.asarray(self.stiffness) @ mapping / self.length_m
        geometric = np.column_stack(
            [
                _map_derivative(
                    relative, mapping @ axis, (inverse_left, inverse_right)
                ).T
                @ resultant
                + _chart_correction(gradient, axis)
                for axis in np.eye(12)
            ]
        )
        return SectionLinearization(
            energy,
            finite_array(gradient, (12,), "section gradient"),
            finite_array(material + geometric, (12, 12), "section tangent"),
            finite_array(material, (12, 12), "material tangent"),
        )


__all__ = ()
