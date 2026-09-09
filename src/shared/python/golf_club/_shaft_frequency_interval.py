"""Conditional inverse bounds between frequency samples of a constant pencil."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._grip_contracts import _finite_norm, finite_array
from ._shaft_damped_spectrum import DampedPencil, _validated_damped_generator
from ._shaft_spectrum import SpectrumScales


class _UnresolvedIntervalError(ValueError):
    """The contraction test cannot qualify this interval; subdivision may help."""


@dataclass(frozen=True)
class FrequencyIntervalControls:
    """Symmetric nonnegative rad/s interval and assumed uniform pencil error.

    The error bounds the spectral norm of additive dynamic-stiffness error in
    the SAME already length-scaled coordinates as the pencil, across the whole
    interval. It is not estimated from calibration or a solve residual. A zero
    error prescribes the nominal finite model. max_contraction lies in (0,1).
    """

    center_rad_s: float
    half_width_rad_s: float
    pencil_error_bound: float
    max_contraction: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(finite_array(getattr(self, name), (), name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        if self.half_width_rad_s > self.center_rad_s:
            raise ValueError("interval must have nonnegative frequency extent")
        if not 0 < self.max_contraction < 1:
            raise ValueError(
                "maximum contraction must lie strictly between zero and one"
            )
        if not math.isfinite(self.center_rad_s + self.half_width_rad_s):
            raise ValueError("frequency interval upper endpoint is nonfinite")


@dataclass(frozen=True)
class FrequencyIntervalAssessment:
    """Floating evaluation of a Neumann bound, not outward-rounded certification.

    Bounds use the spectral norm in the declared length-scaled coordinates.
    Frobenius norms upper-bound the needed operator norms in exact arithmetic.
    The difference is from the computed center inverse X, not from a presumed
    exact center solve. The residual contributes explicitly to contraction.
    Rounding in assembly, norms and final arithmetic is not certified here.
    No nonlinear, stability, modal/mesh or physical-validity claim follows.
    """

    controls: FrequencyIntervalControls
    center_inverse: tuple[tuple[complex, ...], ...]
    center_residual_norm: float
    contraction: float
    inverse_norm_bound: float
    inverse_difference_bound: float

    def inverse_array(self) -> np.ndarray:
        """Return a fresh copy of the exact computed center X used by the bound."""
        return np.array(self.center_inverse, dtype=complex)

    @property
    def evidence_status(self) -> str:
        return "conditional-numerical"

    @property
    def stability_status(self) -> str:
        return "unqualified"


def _norm(matrix: np.ndarray) -> float:
    """Scaled hypot avoids squaring tiny/large entries in the Frobenius norm."""
    return _finite_norm(matrix, "frequency interval norm")


def _product(*values: float) -> float:
    result = math.prod(values)
    if not math.isfinite(result) or (
        result == 0 and all(value > 0 for value in values)
    ):
        raise ValueError(
            "frequency interval positive product is numerically unresolved"
        )
    return result


def _interval_assessment(
    coefficients: tuple[np.ndarray, ...], controls: FrequencyIntervalControls
) -> FrequencyIntervalAssessment:
    mass, gyro, damping, stiffness = coefficients
    frequency, width = controls.center_rad_s, controls.half_width_rad_s
    dynamic = (
        stiffness
        - frequency * (frequency * mass)
        + 1j * frequency * gyro
        + 1j * frequency * damping
    )
    identity = np.eye(len(mass))
    inverse = np.linalg.solve(dynamic, identity)
    residual = _norm(identity - inverse @ dynamic)
    slope = inverse @ (-2 * frequency * mass + 1j * gyro + 1j * damping)
    inverse_norm = _norm(inverse)
    contraction = math.fsum(
        (
            residual,
            _product(width, _norm(slope)),
            _product(width, width, _norm(inverse @ mass)),
            _product(inverse_norm, controls.pencil_error_bound),
        )
    )
    if contraction > controls.max_contraction:
        raise _UnresolvedIntervalError(
            "frequency interval is unresolved at the declared contraction limit"
        )
    bound = inverse_norm / (1 - contraction)
    difference = _product(contraction, bound)
    if not math.isfinite(bound) or bound == 0:
        raise ValueError("frequency interval inverse bound is numerically unresolved")
    return FrequencyIntervalAssessment(
        controls=controls,
        center_inverse=tuple(tuple(complex(item) for item in row) for row in inverse),
        center_residual_norm=residual,
        contraction=contraction,
        inverse_norm_bound=bound,
        inverse_difference_bound=difference,
    )


def assess_frequency_interval(
    pencil: DampedPencil,
    scales: SpectrumScales,
    controls: FrequencyIntervalControls,
) -> FrequencyIntervalAssessment:
    """Bound D(w)^-1 near a center using the full M/G/C/K, or refuse.

    D(w)=K-w^2 M+i*w*G+i*w*C with exp(+i*w*t). Original positive-mass,
    skew-gyro and passive-damping numerical contracts are checked first.
    Nonsymmetric stiffness remains intact. No damping, clipping, inverse
    regularization, basis truncation or autonomous operation is inferred.
    """
    if not isinstance(controls, FrequencyIntervalControls):
        raise TypeError("expected FrequencyIntervalControls")
    _, coefficients = _validated_damped_generator(pencil, scales)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return _interval_assessment(coefficients, controls)
    except (np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
        raise ValueError("frequency interval numerical evaluation failed") from error


__all__ = ()
