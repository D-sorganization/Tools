"""Private frozen spectra retaining gyroscopic transport and damping separately."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_spectrum import (
    FrozenSpectrum,
    SpectrumScales,
    _general_generator,
    _require_skew,
    _spectrum_from_generator,
    _validate_result,
)


@dataclass(frozen=True)
class DampedPencil:
    """Owned finite real square coefficients of M q''+(G+C)q'+Kq=0.

    Stored tuples cannot alias caller arrays. Coefficient shape/finiteness is
    checked here; positive mass, skew G and symmetric semidefinite C are checked
    against the declared numerical tolerances at evaluation. No operating state,
    boundary, autonomy, material identification or stability is inferred.
    """

    mass: object
    gyroscopic: object
    damping: object
    stiffness: object

    def __post_init__(self) -> None:
        shape = np.asarray(self.mass).shape
        if len(shape) != 2 or shape[0] == 0 or shape[0] != shape[1]:
            raise ValueError("mass must be a nonempty square matrix")
        for name in ("mass", "gyroscopic", "damping", "stiffness"):
            matrix = finite_array(getattr(self, name), shape, name)
            object.__setattr__(
                self, name, tuple(tuple(float(x) for x in row) for row in matrix)
            )

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fresh arrays in M/G/C/K order; no hidden combination or projection."""
        return (
            np.array(self.mass),
            np.array(self.gyroscopic),
            np.array(self.damping),
            np.array(self.stiffness),
        )


def _require_damping(damping: np.ndarray, tolerance: float) -> None:
    scale = float(np.linalg.norm(damping))
    if np.linalg.norm(damping - damping.T) > tolerance * scale:
        raise ValueError("damping must be symmetric within the declared tolerance")
    if np.min(np.linalg.eigvalsh(damping)) < -tolerance * scale:
        raise ValueError("damping must be positive semidefinite within tolerance")


def frozen_damped_spectrum(
    pencil: DampedPencil, scales: SpectrumScales
) -> FrozenSpectrum:
    """Compute rates with original-coefficient residuals and eigenbasis condition.

    Already length-scaled coefficients are expected. Tolerances diagnose the
    supplied arrays; they do not repair them or certify exact passivity. Keep
    nonsymmetric stiffness, growing modes, neutral modes and defective bases.
    Frozen eigenvalues alone never establish swing/nonlinear stability.
    """
    if not isinstance(pencil, DampedPencil) or not isinstance(scales, SpectrumScales):
        raise TypeError("expected DampedPencil and SpectrumScales")
    mass, gyro, damping, stiffness = pencil.arrays()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            _require_skew(gyro, scales.residual_tolerance)
            _require_damping(damping, scales.residual_tolerance)
            generator = _general_generator(mass, gyro + damping, stiffness, scales)
            result = _spectrum_from_generator(
                generator, (mass, gyro, damping, stiffness), scales
            )
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("damped spectrum numerical evaluation failed") from error
    _validate_result(result, scales.residual_tolerance)
    return result


__all__ = ()
