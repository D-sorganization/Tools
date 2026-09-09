"""Conditional absolute full/reduced displacement transfer error on an interval."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_frequency_interval import (
    FrequencyIntervalAssessment,
    FrequencyIntervalControls,
    _norm,
    _product,
    assess_frequency_interval,
)
from ._shaft_galerkin import GalerkinReduction
from ._shaft_transfer_polynomial import residual_transfer_bound
from ._shaft_transfer_ports import DisplacementPorts


@dataclass(frozen=True)
class ReductionIntervalControls:
    """Full-pencil interval and additional uniform reduced-pencil error.

    The full error is projected using ||V||_F^2 epsilon. The additional error
    bounds any remaining reduced dynamic-stiffness error in reduced coordinates.
    Both are prescribed spectral-norm bounds over the entire interval, with
    units matching their pencils. Zero prescribes the nominal finite model;
    neither error is estimated from calibration or the inverse residual.
    """

    interval: FrequencyIntervalControls
    additional_reduced_error_bound: float

    def __post_init__(self) -> None:
        if not isinstance(self.interval, FrequencyIntervalControls):
            raise TypeError("expected FrequencyIntervalControls")
        value = float(
            finite_array(
                self.additional_reduced_error_bound, (), "additional reduced error"
            )
        )
        if value < 0:
            raise ValueError("additional reduced error must be nonnegative")
        object.__setattr__(self, "additional_reduced_error_bound", value)


@dataclass(frozen=True)
class ReductionIntervalAssessment:
    """Absolute dimensionless transfer-error bound, conditional on input models.

    For each frequency, ||C Df^-1 B - C V Dr^-1 V.T B||_2 is bounded in exact
    arithmetic by the reported expression. Frobenius norms bound operator
    norms. Actual floating arithmetic is not outward-rounded certification.
    The center error uses computed inverses; both inverse defects remain in
    the bound. No relative, phase, stability or physical-validity claim follows.
    """

    controls: ReductionIntervalControls
    full: FrequencyIntervalAssessment
    reduced: FrequencyIntervalAssessment
    center_error: tuple[tuple[complex, ...], ...]
    absolute_error_bound: float
    inverse_variation_bound: float
    residual_polynomial_bound: float

    def center_error_array(self) -> np.ndarray:
        """Return a fresh full-minus-reduced normalized center transfer."""
        return np.array(self.center_error, dtype=complex)

    @property
    def evidence_status(self) -> str:
        return "conditional-numerical"

    @property
    def stability_status(self) -> str:
        return "unqualified"


def _assess_pair(
    reduction: GalerkinReduction, controls: ReductionIntervalControls
) -> tuple[FrequencyIntervalAssessment, FrequencyIntervalAssessment]:
    basis_norm = _norm(reduction.basis_array())
    projected_error = _product(
        basis_norm, basis_norm, controls.interval.pencil_error_bound
    )
    reduced_error = math.fsum(
        (projected_error, controls.additional_reduced_error_bound)
    )
    reduced_controls = replace(controls.interval, pencil_error_bound=reduced_error)
    full = assess_frequency_interval(
        reduction.full_pencil, reduction.scales, controls.interval
    )
    reduced = assess_frequency_interval(
        reduction.pencil, reduction.scales, reduced_controls
    )
    return full, reduced


def _transfer_bound(
    basis: np.ndarray,
    ports: DisplacementPorts,
    pair: tuple[FrequencyIntervalAssessment, FrequencyIntervalAssessment],
) -> tuple[np.ndarray, float]:
    inputs, outputs = ports.normalized_arrays()
    full, reduced = pair
    center = (
        outputs
        @ (full.inverse_array() - basis @ reduced.inverse_array() @ basis.T)
        @ inputs
    )
    inverse_error = math.fsum(
        (
            full.inverse_difference_bound,
            _product(_norm(basis), _norm(basis), reduced.inverse_difference_bound),
        )
    )
    bound = math.fsum(
        (_norm(center), _product(_norm(outputs), _norm(inputs), inverse_error))
    )
    if not math.isfinite(bound):
        raise ValueError("reduction transfer bound is numerically unresolved")
    return center, bound


def assess_reduction_interval(
    reduction: GalerkinReduction,
    ports: DisplacementPorts,
    controls: ReductionIntervalControls,
) -> ReductionIntervalAssessment:
    """Bound full-minus-Galerkin displacement transfer on the entire interval.

    Require an owned real Galerkin reduction and explicit normalized ports.
    Validate both full and reduced inverses, even for zero observation/loading.
    Singular unobservable states can therefore cause refusal; that limitation
    of this full-inverse method does not prove a divergent port response.
    """
    if not isinstance(reduction, GalerkinReduction) or not isinstance(
        ports, DisplacementPorts
    ):
        raise TypeError("expected GalerkinReduction and DisplacementPorts")
    if not isinstance(controls, ReductionIntervalControls):
        raise TypeError("expected ReductionIntervalControls")
    basis = reduction.basis_array()
    if ports.normalized_arrays()[0].shape[0] != len(basis):
        raise ValueError("port maps must match the full pencil coordinates")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            pair = _assess_pair(reduction, controls)
            center, bound = _transfer_bound(basis, ports, pair)
            polynomial_bound = residual_transfer_bound(
                (reduction.full_pencil, reduction.pencil),
                basis,
                ports.normalized_arrays(),
                pair,
            )
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("reduction transfer numerical evaluation failed") from error
    owned = tuple(tuple(complex(item) for item in row) for row in center)
    return ReductionIntervalAssessment(
        controls, *pair, owned, min(bound, polynomial_bound), bound, polynomial_bound
    )


__all__ = ()
