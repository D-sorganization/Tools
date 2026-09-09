"""Shared work-conjugate length scaling of full-node gripped operators."""

from __future__ import annotations

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_damped_spectrum import DampedPencil
from ._shaft_gripped_dynamics import GrippedChainDynamics
from ._shaft_spectrum import SpectrumScales


def scaled_gripped_operators(
    operators: GrippedChainDynamics, scales: SpectrumScales
) -> tuple[DampedPencil, np.ndarray, np.ndarray]:
    """Return S.T operator S, diag(S), and S.T r without dropping any node.

    Require complete six-axis nodes and finite real coefficients. Refuse
    unrepresentable scaling, including underflow, without repairing the plant.
    Positive-mass and decay/spectral qualification remain evaluation contracts.
    """
    if not isinstance(operators, GrippedChainDynamics) or not isinstance(
        scales, SpectrumScales
    ):
        raise TypeError("expected GrippedChainDynamics and SpectrumScales")
    shape = np.shape(operators.mass)
    if len(shape) != 2 or shape[0] == 0 or shape[0] != shape[1] or shape[0] % 6:
        raise ValueError("gripped operators require complete six-axis nodes")
    coordinates = np.tile([scales.length_m] * 3 + [1.0] * 3, shape[0] // 6)
    matrices = tuple(
        finite_array(getattr(operators, name), shape, name)
        for name in ("mass", "gyroscopic", "damping", "stiffness")
    )
    residual = finite_array(operators.residual, (shape[0],), "residual")
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            congruence = coordinates[:, None] * coordinates[None, :]
            pencil = DampedPencil(*(matrix * congruence for matrix in matrices))
            scaled_residual = residual * coordinates
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("gripped coordinate scaling is not representable") from error
    return pencil, coordinates, scaled_residual


__all__ = ()
