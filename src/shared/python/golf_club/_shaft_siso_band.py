"""Complete SISO bands qualified against absolute, relative and phase targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

from ._grip_contracts import finite_array
from ._shaft_frequency_band import _assess_cover
from ._shaft_frequency_interval import (
    FrequencyIntervalControls,
    _UnresolvedIntervalError,
)
from ._shaft_galerkin import GalerkinReduction
from ._shaft_reduction_band import ReductionBandControls
from ._shaft_reduction_interval import ReductionIntervalControls
from ._shaft_siso_interval import (
    SisoReductionIntervalAssessment,
    assess_siso_reduction_interval,
)
from ._shaft_transfer_ports import DisplacementPorts


@dataclass(frozen=True)
class SisoReductionBandControls:
    """Existing absolute-band prescription plus strict relative and phase targets.

    Relative complex/magnitude error lies in (0,1); the principal phase error
    target lies in (0,pi/2) radians. Every interval must meet all three targets.
    All original uncertainty, normalization and attempted-pair budget contracts
    remain attached to the absolute-band prescription.
    """

    absolute: ReductionBandControls
    maximum_relative_error: float
    maximum_phase_error_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.absolute, ReductionBandControls):
            raise TypeError("expected ReductionBandControls")
        for name, limit in (
            ("maximum_relative_error", 1.0),
            ("maximum_phase_error_rad", math.pi / 2),
        ):
            value = float(finite_array(getattr(self, name), (), name))
            if not 0 < value < limit:
                raise ValueError(f"{name} must lie strictly between zero and {limit}")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SisoReductionBandCell:
    """A closed cell within its common full/reduced assessment interval."""

    lower_rad_s: float
    upper_rad_s: float
    assessment: SisoReductionIntervalAssessment


@dataclass(frozen=True)
class SisoReductionBandAssessment:
    """Complete cover with nonzero full-response floors and bounded phase ratios.

    Every cell meets the absolute, relative complex/magnitude and principal
    phase-error targets. These are conditional floating finite-model bounds;
    no global phase unwrapping, acoustic calibration or stability is inferred.
    """

    controls: SisoReductionBandControls
    cells: tuple[SisoReductionBandCell, ...]
    evaluation_count: int
    evidence_status: ClassVar[str] = "conditional-numerical"
    stability_status: ClassVar[str] = "unqualified"

    @property
    def minimum_full_response_lower_bound(self) -> float:
        return float(
            min(cell.assessment.full_response_lower_bound for cell in self.cells)
        )

    @property
    def maximum_relative_complex_error_bound(self) -> float:
        return float(
            max(cell.assessment.relative_complex_error_bound for cell in self.cells)
        )

    @property
    def maximum_relative_magnitude_error_bound(self) -> float:
        return self.maximum_relative_complex_error_bound

    @property
    def maximum_phase_error_bound_rad(self) -> float:
        return float(max(cell.assessment.phase_error_bound_rad for cell in self.cells))


def _require_targets(
    result: SisoReductionIntervalAssessment,
    controls: SisoReductionBandControls,
) -> None:
    errors = (
        result.reduction.absolute_error_bound,
        result.relative_complex_error_bound,
        result.phase_error_bound_rad,
    )
    limits = (
        controls.absolute.maximum_absolute_error,
        controls.maximum_relative_error,
        controls.maximum_phase_error_rad,
    )
    if any(error > limit for error, limit in zip(errors, limits, strict=True)):
        raise _UnresolvedIntervalError("SISO reduction error targets are unresolved")


def assess_siso_reduction_band(
    reduction: GalerkinReduction,
    ports: DisplacementPorts,
    controls: SisoReductionBandControls,
) -> SisoReductionBandAssessment:
    """Cover the entire closed band or refuse without returning partial results."""
    if not isinstance(controls, SisoReductionBandControls):
        raise TypeError("expected SisoReductionBandControls")

    def accepted(
        interval: FrequencyIntervalControls,
    ) -> SisoReductionIntervalAssessment:
        paired = ReductionIntervalControls(
            interval, controls.absolute.additional_reduced_error_bound
        )
        result = assess_siso_reduction_interval(reduction, ports, paired)
        _require_targets(result, controls)
        return result

    cells, evaluations = _assess_cover(controls.absolute.band, accepted)
    return SisoReductionBandAssessment(
        controls, tuple(SisoReductionBandCell(*cell) for cell in cells), evaluations
    )


__all__ = ()
