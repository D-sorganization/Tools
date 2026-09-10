"""Nonzero-response floors and pointwise SISO reduction magnitude/phase bounds."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ._shaft_frequency_interval import _norm, _product, _UnresolvedIntervalError
from ._shaft_galerkin import GalerkinReduction
from ._shaft_reduction_interval import (
    ReductionIntervalAssessment,
    ReductionIntervalControls,
    assess_reduction_interval,
)
from ._shaft_transfer_polynomial import _response_polynomial
from ._shaft_transfer_ports import DisplacementPorts


def _minimum_segment_modulus(center: complex, span: complex) -> float:
    """Minimum |center+x span| for real x in [-1,1], using scaled geometry.

    Project the origin onto the line and clip to its endpoints. Normalize
    before projection, avoiding squared norms of huge/tiny complex values.
    This is floating geometry, not outward-rounded interval arithmetic.
    """
    center_value, span_value = np.complex128(center), np.complex128(span)
    center_norm, span_norm = (
        _norm(np.asarray(center_value)),
        _norm(np.asarray(span_value)),
    )
    if span_norm == 0:
        return float(center_norm)
    scale = max(center_norm, span_norm)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        normalized = center_value / scale
        direction = span_value / span_norm
        extent = float(np.divide(span_norm, scale))
        projection = float((normalized * direction.conjugate()).real)
        position = min(extent, max(-extent, -projection))
        distance = _norm(np.asarray(normalized + position * direction))
    return float(_product(scale, distance))


def _siso_response_floor(
    reduction: GalerkinReduction,
    ports: DisplacementPorts,
    assessment: ReductionIntervalAssessment,
) -> float:
    """Subtract the full response's bounded residual error from its line minimum."""
    inputs, outputs = ports.normalized_arrays()
    polynomial = _response_polynomial(reduction.full_pencil, assessment.full, inputs)
    center = (outputs @ polynomial.constant)[0, 0]
    interval = assessment.controls.interval
    width = interval.half_width_rad_s
    span = width * (outputs @ polynomial.linear)[0, 0]
    minimum = _minimum_segment_modulus(complex(center), complex(span))
    remainder = _product(_norm(outputs), polynomial.remainder_bound)
    return float(max(0.0, minimum - remainder))


def _relative_error(error: float, floor: float) -> float:
    if floor <= 0:
        raise _UnresolvedIntervalError("full response floor is not strictly positive")
    relative = float(np.divide(error, floor))
    if not math.isfinite(relative) or (error > 0 and relative == 0):
        raise ValueError("relative error is numerically unresolved")
    if relative >= 1:
        raise _UnresolvedIntervalError("relative error cannot qualify phase below pi/2")
    return relative


@dataclass(frozen=True)
class SisoReductionIntervalAssessment:
    """Conditional full-response floor and relative complex/magnitude/phase errors.

    For every frequency in the interval, |Hfull| is at least the positive
    dimensionless floor and |Hreduced/Hfull-1| <= relative_complex_error_bound
    < 1. The relative magnitude error is at most the same value. The principal
    phase of Hreduced/Hfull has magnitude <= asin(relative error), in radians.
    Neither response can vanish under these conditional bounds.

    This does not assign phase to a MIMO matrix, unwrap absolute phase across
    frequency, establish time-domain stability, or qualify measured acoustics.
    Floating calculations are not certified outward-rounded bounds.
    """

    reduction: ReductionIntervalAssessment
    full_response_lower_bound: float
    relative_complex_error_bound: float
    phase_error_bound_rad: float
    evidence_status: ClassVar[str] = "conditional-numerical"
    stability_status: ClassVar[str] = "unqualified"

    @property
    def relative_magnitude_error_bound(self) -> float:
        return self.relative_complex_error_bound


def assess_siso_reduction_interval(
    reduction: GalerkinReduction,
    ports: DisplacementPorts,
    controls: ReductionIntervalControls,
) -> SisoReductionIntervalAssessment:
    """Require a nonzero full-response floor before forming relative/phase bounds.

    Preconditions retain the existing full/reduced pencil, uncertainty and
    normalization contracts. Exactly one input and one output are required.
    Refusal to resolve a floor or a phase below pi/2 can trigger subdivision;
    it is not a proof that an actual response vanishes or the model is invalid.
    """
    if not isinstance(ports, DisplacementPorts):
        raise TypeError("expected DisplacementPorts")
    inputs, outputs = ports.normalized_arrays()
    if inputs.shape[1] != 1 or outputs.shape[0] != 1:
        raise ValueError("phase requires a single input and single output")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            assessed = assess_reduction_interval(reduction, ports, controls)
            floor = _siso_response_floor(reduction, ports, assessed)
            relative = _relative_error(assessed.absolute_error_bound, floor)
            phase = math.asin(relative)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("SISO response numerical evaluation failed") from error
    return SisoReductionIntervalAssessment(assessed, floor, relative, phase)


__all__ = ()
