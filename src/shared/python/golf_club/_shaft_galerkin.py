"""Explicit real Galerkin subspaces for the existing second-order shaft pencil."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_damped_spectrum import DampedPencil, _validated_damped_generator
from ._shaft_spectrum import SpectrumScales


def _project_pencil(
    full: DampedPencil, basis: np.ndarray, scales: SpectrumScales
) -> DampedPencil:
    """Use one congruence for each coefficient, without symmetry repair."""
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            reduced = DampedPencil(
                *(basis.T @ matrix @ basis for matrix in full.arrays())
            )
            _validated_damped_generator(reduced, scales)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("Galerkin projection numerical evaluation failed") from error
    return reduced


def _mapped_vector(matrix: np.ndarray, value: object, name: str) -> tuple[float, ...]:
    vector = finite_array(value, (matrix.shape[1],), name)
    try:
        with np.errstate(over="raise", invalid="raise"):
            mapped = finite_array(matrix @ vector, (matrix.shape[0],), name)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError(f"{name} mapping numerical evaluation failed") from error
    return tuple(float(item) for item in mapped)


@dataclass(frozen=True)
class GalerkinReduction:
    """Owned constant real basis y=V z in already length-scaled coordinates.

    The original and reduced pencils must both satisfy the existing numerical
    mass/gyro/damping domain. Positive resolved reduced mass checks weighted
    basis independence. Nonsymmetric K and separate G/C are retained. No
    orthonormalization, mass lumping, regularization or mode selection occurs.

    This is a subspace approximation, not a statement that discarded motion,
    loads or instability are absent. It supplies no error, frequency-band,
    nonlinear, acoustic or stability certificate. Basis selection and full vs
    reduced response/convergence evidence remain the caller's responsibility.
    """

    full_pencil: DampedPencil
    basis: object
    scales: SpectrumScales
    pencil: DampedPencil = field(init=False)

    def __post_init__(self) -> None:
        _validated_damped_generator(self.full_pencil, self.scales)
        size = len(self.full_pencil.mass)
        shape = np.asarray(self.basis).shape
        if len(shape) != 2 or shape[0] != size or not 1 <= shape[1] <= size:
            raise ValueError("basis must have N rows and between one and N columns")
        basis = finite_array(self.basis, shape, "Galerkin basis")
        reduced = _project_pencil(self.full_pencil, basis, self.scales)
        owned = tuple(tuple(float(item) for item in row) for row in basis)
        object.__setattr__(self, "basis", owned)
        object.__setattr__(self, "pencil", reduced)

    def basis_array(self) -> np.ndarray:
        """Return a fresh V for real or complex response/observation mapping."""
        return np.array(self.basis)

    def reduce_force(self, full_force: object) -> tuple[float, ...]:
        """Return V.T f for a finite real force or left-side residual.

        The force is conjugate to length-scaled y, not an unscaled SI wrench.
        The same transpose preserves force-velocity work and residual sign.
        A force orthogonal to V maps to zero; this does not bound its omitted
        full-system response. It is not a displacement/initial-state projection.
        """
        return _mapped_vector(self.basis_array().T, full_force, "full force")

    def lift_motion(self, reduced_motion: object) -> tuple[float, ...]:
        """Return V z for finite real displacement, velocity or acceleration.

        Also applies to the T-scaled velocity half of the existing state.
        Input and output must share the declared coordinate/time convention.
        No inverse mapping or inferred discarded initial motion is supplied.
        """
        return _mapped_vector(self.basis_array(), reduced_motion, "reduced motion")

    @property
    def stability_status(self) -> str:
        """Truncation can hide growing directions; no stability is inferred."""
        return "unqualified"


__all__ = ()
