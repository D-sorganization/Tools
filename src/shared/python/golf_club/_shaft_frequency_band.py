"""Conditional complete-band bounds composed from the existing interval test."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from typing import TypeVar

import numpy as np

from ._grip_contracts import finite_array
from ._shaft_damped_spectrum import DampedPencil
from ._shaft_frequency_interval import (
    FrequencyIntervalAssessment,
    FrequencyIntervalControls,
    _UnresolvedIntervalError,
    assess_frequency_interval,
)
from ._shaft_spectrum import SpectrumScales

_Assessment = TypeVar("_Assessment")


@dataclass(frozen=True)
class FrequencyBandControls:
    """Closed nonnegative rad/s band, uniform pencil error and finite work budget.

    Error and contraction retain FrequencyIntervalControls' units and meaning.
    The evaluation budget counts attempted interval assessments, including
    rejected parent cells. It never authorizes returning an incomplete cover.
    """

    bounds_rad_s: tuple[float, float]
    pencil_error_bound: float
    max_contraction: float
    max_evaluations: int

    def __post_init__(self) -> None:
        bounds = finite_array(self.bounds_rad_s, (2,), "frequency bounds")
        if not 0 <= bounds[0] <= bounds[1]:
            raise ValueError("frequency bounds must be ordered and nonnegative")
        object.__setattr__(self, "bounds_rad_s", tuple(float(x) for x in bounds))
        scalar = FrequencyIntervalControls(
            0, 0, self.pencil_error_bound, self.max_contraction
        )
        object.__setattr__(self, "pencil_error_bound", scalar.pencil_error_bound)
        object.__setattr__(self, "max_contraction", scalar.max_contraction)
        budget: object = self.max_evaluations
        if isinstance(budget, (bool, np.bool_)) or not isinstance(
            budget, (int, np.integer)
        ):
            raise TypeError("frequency evaluation budget must be an integer")
        if budget < 1:
            raise ValueError("frequency evaluation budget must be positive")
        object.__setattr__(self, "max_evaluations", int(budget))


@dataclass(frozen=True)
class FrequencyBandCell:
    """Nominal closed cell contained within the assessment's symmetric interval."""

    lower_rad_s: float
    upper_rad_s: float
    assessment: FrequencyIntervalAssessment


@dataclass(frozen=True)
class FrequencyBandAssessment:
    """Complete ordered cover, conditional on the declared pencil and error.

    Exact binary endpoint bookkeeping prevents gaps; it does not certify the
    floating matrix assembly, norms or Neumann arithmetic. No stability,
    modal/mesh convergence, nonlinear or physical bandwidth is inferred.
    """

    controls: FrequencyBandControls
    cells: tuple[FrequencyBandCell, ...]
    evaluation_count: int

    @property
    def maximum_inverse_norm_bound(self) -> float:
        return max(cell.assessment.inverse_norm_bound for cell in self.cells)

    @property
    def evidence_status(self) -> str:
        return "conditional-numerical"

    @property
    def stability_status(self) -> str:
        return "unqualified"


def _ceil_float(value: Fraction) -> float:
    rounded = float(value)
    return math.nextafter(rounded, math.inf) if Fraction(rounded) < value else rounded


def _cover_interval(
    lower: float, upper: float, controls: FrequencyBandControls
) -> FrequencyIntervalControls:
    """Enclose exact binary endpoints despite rounded midpoint/radius arithmetic."""
    left, right = Fraction(lower), Fraction(upper)
    center = _ceil_float((left + right) / 2)
    midpoint = Fraction(center)
    width = _ceil_float(max(midpoint - left, right - midpoint))
    return FrequencyIntervalControls(
        center, width, controls.pencil_error_bound, controls.max_contraction
    )


def assess_frequency_band(
    pencil: DampedPencil,
    scales: SpectrumScales,
    controls: FrequencyBandControls,
) -> FrequencyBandAssessment:
    """Cover the entire closed band or raise; never return a partial assessment.

    Only an unresolved contraction triggers subdivision. Invalid plants,
    singular solves and other numerical failures propagate without repairs.
    Every split has a representable interior boundary shared by both children.
    Returned bounds describe the supplied finite model, not a measured club.
    """
    if not isinstance(controls, FrequencyBandControls):
        raise TypeError("expected FrequencyBandControls")
    cells, evaluations = _assess_cover(
        controls, partial(assess_frequency_interval, pencil, scales)
    )
    return FrequencyBandAssessment(
        controls, tuple(FrequencyBandCell(*cell) for cell in cells), evaluations
    )


def _assess_cover(
    controls: FrequencyBandControls,
    assess: Callable[[FrequencyIntervalControls], _Assessment],
) -> tuple[tuple[tuple[float, float, _Assessment], ...], int]:
    """Share exact endpoint coverage, attempted-cell budgets and fail-closed splits."""
    pending = [controls.bounds_rad_s]
    cells: list[tuple[float, float, _Assessment]] = []
    evaluations = 0
    while pending:
        if evaluations >= controls.max_evaluations:
            raise ValueError("frequency band evaluation budget exhausted")
        lower, upper = pending.pop()
        interval = _cover_interval(lower, upper, controls)
        evaluations += 1
        try:
            assessment = assess(interval)
        except _UnresolvedIntervalError:
            midpoint = interval.center_rad_s
            if not lower < midpoint < upper:
                raise ValueError(
                    "frequency band has no representable interior subdivision"
                ) from None
            pending.extend(((midpoint, upper), (lower, midpoint)))
        else:
            cells.append((lower, upper, assessment))
    return tuple(cells), evaluations


__all__ = ()
