"""Residual-corrected local transfer approximations with explicit remainders."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._shaft_damped_spectrum import DampedPencil
from ._shaft_frequency_interval import FrequencyIntervalAssessment, _norm, _product


@dataclass(frozen=True)
class _ResponsePolynomial:
    """P0+delta P1 with a uniform bound on ||D^-1 B-P|| over the interval."""

    constant: np.ndarray
    linear: np.ndarray
    remainder_bound: float


def _response_polynomial(
    pencil: DampedPencil,
    assessment: FrequencyIntervalAssessment,
    inputs: np.ndarray,
) -> _ResponsePolynomial:
    """Keep computed-center residuals; no Taylor remainder is discarded."""
    mass, gyro, damping, stiffness = pencil.arrays()
    controls = assessment.controls
    frequency, width = controls.center_rad_s, controls.half_width_rad_s
    dynamic = (
        stiffness
        - frequency * (frequency * mass)
        + 1j * frequency * gyro
        + 1j * frequency * damping
    )
    slope = -2 * frequency * mass + 1j * gyro + 1j * damping
    inverse = assessment.inverse_array()
    constant = inverse @ inputs
    linear = -inverse @ slope @ constant
    # D2=-M, so the cubic residual's last two signs are positive for M terms.
    residuals = (
        inputs - dynamic @ constant,
        -dynamic @ linear - slope @ constant,
        -slope @ linear + mass @ constant,
        mass @ linear,
    )
    residual_bound = math.fsum(
        _product(*([width] * degree), _norm(value))
        for degree, value in enumerate(residuals)
    )
    polynomial_bound = math.fsum((_norm(constant), _product(width, _norm(linear))))
    uncertain_residual = math.fsum(
        (residual_bound, _product(controls.pencil_error_bound, polynomial_bound))
    )
    remainder = _product(assessment.inverse_norm_bound, uncertain_residual)
    return _ResponsePolynomial(constant, linear, remainder)


def residual_transfer_bound(
    pencils: tuple[DampedPencil, DampedPencil],
    basis: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray],
    pair: tuple[FrequencyIntervalAssessment, FrequencyIntervalAssessment],
) -> float:
    """Bound the output polynomial difference plus both inverse residual errors.

    Each actual response equals its polynomial plus D^-1(B-DP), including
    declared dynamic-stiffness uncertainty. Comparing full/lifted polynomials
    first preserves shared center/slope cancellation. This remains conditional
    floating evaluation of an exact-arithmetic bound, not a fitted estimator.
    """
    inputs, outputs = maps
    full = _response_polynomial(pencils[0], pair[0], inputs)
    reduced = _response_polynomial(pencils[1], pair[1], basis.T @ inputs)
    observed_basis = outputs @ basis
    center = outputs @ full.constant - observed_basis @ reduced.constant
    slope = outputs @ full.linear - observed_basis @ reduced.linear
    width = pair[0].controls.half_width_rad_s
    return math.fsum(
        (
            _norm(center),
            _product(width, _norm(slope)),
            _product(_norm(outputs), full.remainder_bound),
            _product(_norm(observed_basis), reduced.remainder_bound),
        )
    )


__all__ = ()
