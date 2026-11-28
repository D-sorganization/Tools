"""Centered exponential trendline shared by plot and data-explorer fits.

Calendar-year abscissas make ``a * exp(b * x)`` overflow: the uncentered
scale underflows to 0 and the product is NaN. The fit is performed in
shifted coordinates and folded back only when that form stays finite on the
points the caller will publish.

Numpy only. Callers that still need a nonlinear refine (``curve_fit``) do
that themselves and pass the refined scale and rate back through
:func:`stabilize_exponential`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# math.exp overflows a float64 past about 709.
_MAX_EXPONENT = 700.0


@dataclass(frozen=True)
class ExponentialModel:
    """``y = scale * exp(rate * (x - origin))``.

    When ``origin_in_equation`` is false, ``origin`` is 0 and ``scale`` is the
    uncentered factor in ``y = scale * exp(rate * x)``.
    """

    scale: float
    rate: float
    origin: float
    origin_in_equation: bool

    def coefficients(self) -> list[float]:
        """``[scale, rate]``, plus ``origin`` when the equation keeps the shift."""
        if self.origin_in_equation:
            return [self.scale, self.rate, self.origin]
        return [self.scale, self.rate]

    def equation(self, *, precision: str, spaced: bool) -> str:
        """Render the published equation.

        ``precision`` is a format spec such as ``.4g`` or ``.6g``. ``spaced``
        selects ``exp(b * x)`` (data explorer) over ``exp(bx)`` (plot engine).
        """
        scale = format(self.scale, precision)
        rate = format(self.rate, precision)
        if self.origin_in_equation:
            origin = format(self.origin, precision)
            if spaced:
                return f"y = {scale} * exp({rate} * (x - {origin}))"
            return f"y = {scale} * exp({rate}(x - {origin}))"
        if spaced:
            return f"y = {scale} * exp({rate} * x)"
        return f"y = {scale} * exp({rate}x)"

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the model. Preconditions: ``x`` is finite."""
        shifted = np.asarray(x, dtype=np.float64) - self.origin
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = self.scale * np.exp(self.rate * shifted)
        return predicted


def fit_log_linear_exponential(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    """Return ``(scale, rate, origin)`` for a centered log-linear exponential.

    The model is ``y = scale * exp(rate * (x - origin))`` with ``origin``
    equal to the mean of ``x``.

    Preconditions: ``x`` and ``y`` are finite, equal-length, 1-D, length at
    least 2, and every ``y`` is > 0.

    Postcondition: ``scale``, ``rate``, and ``origin`` are finite and
    ``scale`` > 0.
    """
    abscissa = np.asarray(x, dtype=np.float64)
    ordinate = np.asarray(y, dtype=np.float64)
    if abscissa.ndim != 1 or ordinate.ndim != 1 or abscissa.shape != ordinate.shape:
        raise ValueError("x and y must be equal-length 1-D arrays")
    if abscissa.size < 2:
        raise ValueError("exponential fit requires at least 2 points")
    if not np.all(np.isfinite(abscissa)) or not np.all(np.isfinite(ordinate)):
        raise ValueError("exponential fit inputs must be finite")
    if np.any(ordinate <= 0.0):
        raise ValueError("exponential fit requires all y > 0")

    origin = float(np.mean(abscissa))
    slope_intercept = np.polyfit(abscissa - origin, np.log(ordinate), 1)
    rate = float(slope_intercept[0])
    scale = float(np.exp(slope_intercept[1]))
    if not (
        math.isfinite(scale)
        and math.isfinite(rate)
        and math.isfinite(origin)
        and scale > 0.0
    ):
        raise ValueError("exponential fit failed to produce a finite centered scale")
    return scale, rate, origin


def stabilize_exponential(
    scale: float, rate: float, origin: float, x_check: np.ndarray
) -> ExponentialModel:
    """Fold ``origin`` into ``scale`` only when ``a*exp(b*x)`` stays finite.

    Preconditions: ``scale > 0`` and ``scale``, ``rate``, ``origin``, and
    ``x_check`` are finite. ``x_check`` is non-empty.

    Postcondition: :meth:`ExponentialModel.predict` on ``x_check`` is finite.
    """
    if not (math.isfinite(scale) and math.isfinite(rate) and math.isfinite(origin)):
        raise ValueError("exponential parameters must be finite")
    if scale <= 0.0:
        raise ValueError("exponential scale must be > 0")
    check = np.asarray(x_check, dtype=np.float64)
    if check.size == 0 or not np.all(np.isfinite(check)):
        raise ValueError("exponential check points must be finite and non-empty")

    exponent = -rate * origin
    uncentered: float | None
    if math.isfinite(exponent) and abs(exponent) <= _MAX_EXPONENT:
        uncentered = scale * math.exp(exponent)
    else:
        uncentered = None

    if uncentered is not None and math.isfinite(uncentered) and uncentered != 0.0:
        candidate = ExponentialModel(uncentered, rate, 0.0, False)
        if np.all(np.isfinite(candidate.predict(check))):
            return candidate
    centered = ExponentialModel(scale, rate, origin, True)
    if not np.all(np.isfinite(centered.predict(check))):
        raise ValueError("exponential fit produced non-finite predictions")
    return centered
