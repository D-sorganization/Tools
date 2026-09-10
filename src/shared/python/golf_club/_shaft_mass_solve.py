"""Shared scaled positive-mass solve for material, linear-first body twists."""

from __future__ import annotations

import numpy as np

from ._grip_contracts import _finite_norm, finite_array
from ._shaft_spectrum import SpectrumScales, _validated_mass


def solve_material_mass(
    mass: object, right_hand_side: object, scales: SpectrumScales
) -> tuple[np.ndarray, float]:
    """Solve M a = f with the shaft's explicit scales and residual refusal.

    Blocks are linear-first six-vectors. No inertia projection, truncation or
    regularization is performed. This extracts the moving-chain solve without
    changing its numerical thresholds; it also serves a single free body.
    """
    if not isinstance(scales, SpectrumScales):
        raise TypeError("scales must be SpectrumScales")
    shape = np.asarray(mass).shape
    if len(shape) != 2 or shape[0] == 0 or shape[0] % 6 or shape[0] != shape[1]:
        raise ValueError("mass must be square with complete six-axis body blocks")
    size = shape[0]
    source = finite_array(mass, shape, "material mass")
    rhs = finite_array(right_hand_side, (size,), "material right hand side")
    coordinates = np.tile([scales.length_m] * 3 + [1.0] * 3, size // 6)
    with np.errstate(under="raise"):
        scaled_mass = source * coordinates[:, None] * coordinates[None, :]
        force = rhs * coordinates
    _validated_mass(scaled_mass, scales)
    acceleration = np.linalg.solve(scaled_mass, force)
    defect = _finite_norm(scaled_mass @ acceleration - force, "acceleration residual")
    denominator = _finite_norm(scaled_mass, "mass") * _finite_norm(
        acceleration, "acceleration"
    ) + _finite_norm(force, "force")
    residual = float(defect / denominator) if denominator > 0 else float(defect)
    if (
        not np.isfinite(denominator)
        or not np.isfinite(residual)
        or residual > scales.residual_tolerance
    ):
        raise ValueError("material acceleration residual is unresolved")
    rates = finite_array(coordinates * acceleration, (size,), "body twist rates")
    return rates, residual


__all__ = ()
