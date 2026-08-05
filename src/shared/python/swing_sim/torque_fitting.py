"""Validated polynomial fitting for sampled SI joint-torque histories."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike, NDArray

from shared.python.contracts import require

from .torque_profiles import FitMetadata, TorquePolynomial


def _fit_arrays(
    times_s: ArrayLike, torque_nm: ArrayLike, degree: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and normalize sampled fit inputs."""
    require(
        type(degree) is int and degree >= 0,
        "degree must be an integer >= 0",
    )
    times = np.asarray(times_s, dtype=np.float64)
    torques = np.asarray(torque_nm, dtype=np.float64)
    require(
        times.ndim == 1 and torques.ndim == 1,
        "fit samples must be one-dimensional",
    )
    require(
        times.shape == torques.shape,
        "time and torque samples must have equal shape",
    )
    require(
        len(times) >= 2 and len(times) > degree,
        "insufficient samples for degree",
    )
    require(np.all(np.isfinite(times)), "time samples must be finite")
    require(np.all(np.isfinite(torques)), "torque samples must be finite")
    require(
        np.all(np.diff(times) > 0.0),
        "time samples must be strictly increasing",
    )
    return times, torques


def _sample_sha256(times: NDArray[np.float64], torques: NDArray[np.float64]) -> str:
    """Hash canonical little-endian float64 time/torque sample pairs."""
    samples = np.column_stack((times, torques)).astype("<f8", copy=False)
    return hashlib.sha256(samples.tobytes(order="C")).hexdigest()


def _r_squared(torques: NDArray[np.float64], residuals: NDArray[np.float64]) -> float:
    """Calculate R squared with a defined result for constant histories."""
    ss_residual = float(np.dot(residuals, residuals))
    centered = torques - float(np.mean(torques))
    ss_total = float(np.dot(centered, centered))
    if ss_total > 0.0:
        return 1.0 - ss_residual / ss_total
    return 1.0 if ss_residual == 0.0 else 0.0


def _condition_number(times: NDArray[np.float64], degree: int) -> float:
    """Condition number of the normalized ascending Vandermonde matrix."""
    normalized = 2.0 * (times - times[0]) / (times[-1] - times[0]) - 1.0
    vandermonde = np.polynomial.polynomial.polyvander(normalized, degree)
    return float(np.linalg.cond(vandermonde))


def fit_torque_polynomial(
    times_s: ArrayLike, torque_nm: ArrayLike, degree: int
) -> TorquePolynomial:
    """Fit ascending coefficients in physical seconds.

    The least-squares solve uses normalized time for conditioning and then
    converts the result back to the ordinary physical-time power basis.
    """
    times, torques = _fit_arrays(times_s, torque_nm, degree)
    fitted = Polynomial.fit(times, torques, degree, domain=[times[0], times[-1]])
    physical = fitted.convert()
    predicted = np.asarray(physical(times), dtype=np.float64)
    residuals = torques - predicted
    metadata = FitMetadata(
        degree=degree,
        rmse_nm=float(np.sqrt(np.mean(residuals**2))),
        max_abs_error_nm=float(np.max(np.abs(residuals))),
        r_squared=_r_squared(torques, residuals),
        condition_number=_condition_number(times, degree),
        original_sample_sha256=_sample_sha256(times, torques),
    )
    coefficients = tuple(float(value) for value in physical.coef)
    return TorquePolynomial(coefficients, metadata)


__all__ = ["fit_torque_polynomial"]
