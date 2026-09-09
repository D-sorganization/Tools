"""Finite-time motion of an explicitly assumed constant affine shaft model."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ._grip_contracts import finite_array
from ._shaft_affine_response import _nominal_input
from ._shaft_damped_spectrum import DampedPencil, _validated_damped_generator
from ._shaft_spectrum import SpectrumScales


def _affine_exponent(
    generator: np.ndarray, input_per_s: np.ndarray, time_s: float, scale_s: float
) -> np.ndarray:
    """Form the augmented exponent without requiring an invertible generator."""
    size = len(generator)
    augmented = np.zeros((size + 1, size + 1))
    elapsed_scaled = time_s / scale_s
    if elapsed_scaled == 0:
        raise ValueError("positive scaled elapsed time is unresolved by underflow")
    # Underflow while scaling a nonzero coefficient loses the supplied model.
    # Zero elapsed time is handled before this helper is called.
    with np.errstate(under="raise"):
        augmented[:size, :size] = generator * elapsed_scaled
        augmented[:size, size] = input_per_s * time_s
    return finite_array(augmented, augmented.shape, "affine exponent")


def affine_state_at(
    pencil: DampedPencil,
    residual: object,
    scales: SpectrumScales,
    initial_state: object,
    time_s: object,
) -> tuple[float, ...]:
    """Evaluate r+M ydd+(G+C)yd+K y=0 at nonnegative physical time.

    Coefficients/residual are already length-scaled. Input/output states are
    x=(y,T*ydot), at time zero and elapsed seconds respectively. Reuse the
    existing positive-mass, skew-G, passive-C domain; retain nonsymmetric K,
    neutral/defective modes and finite growing motion. The caller prescribes
    constant coefficients and load, not merely a frozen swing snapshot.

    Return an owned finite tuple; invalid/unrepresentable evaluation raises.
    The matrix exponential is evaluated numerically, without an eigenbasis or
    a time-step approximation. It supplies no certified forward-error bound,
    nonlinear strain-domain guarantee, stability or physical qualification.
    """
    generator, coefficients = _validated_damped_generator(pencil, scales)
    size = len(generator)
    state = finite_array(initial_state, (size,), "initial scaled state")
    load = finite_array(residual, (size // 2,), "scaled residual")
    elapsed = float(finite_array(time_s, (), "elapsed time"))
    if elapsed < 0:
        raise ValueError("elapsed time must be nonnegative")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            forcing = np.asarray(_nominal_input(coefficients[0], load, scales.time_s))
            if elapsed == 0:
                return tuple(float(value) for value in state)
            exponent = _affine_exponent(generator, forcing, elapsed, scales.time_s)
            propagated = expm(exponent) @ np.r_[state, 1.0]
            result = finite_array(propagated[:size], (size,), "propagated scaled state")
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("affine transient numerical evaluation failed") from error
    return tuple(float(value) for value in result)


__all__ = ()
