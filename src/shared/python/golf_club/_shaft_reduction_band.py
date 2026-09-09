"""Complete-band absolute transfer-error acceptance for a Galerkin shaft model."""

from __future__ import annotations

from dataclasses import dataclass

from ._grip_contracts import finite_array
from ._shaft_frequency_band import FrequencyBandControls, _assess_cover
from ._shaft_frequency_interval import (
    FrequencyIntervalControls,
    _UnresolvedIntervalError,
)
from ._shaft_galerkin import GalerkinReduction
from ._shaft_reduction_interval import (
    ReductionIntervalAssessment,
    ReductionIntervalControls,
    assess_reduction_interval,
)
from ._shaft_transfer_ports import DisplacementPorts


@dataclass(frozen=True)
class ReductionBandControls:
    """Complete frequency band, extra reduced uncertainty and absolute error target.

    The positive target is dimensionless after explicit port normalization.
    The band budget counts attempted paired cells, including rejected parents;
    a pair can stop after its full-system assessment fails. No incomplete cover
    or sampled replacement is returned on exhaustion. Reducing cell width can
    resolve conservatism, but cannot eliminate an actual omitted-mode error.
    """

    band: FrequencyBandControls
    additional_reduced_error_bound: float
    maximum_absolute_error: float

    def __post_init__(self) -> None:
        if not isinstance(self.band, FrequencyBandControls):
            raise TypeError("expected FrequencyBandControls")
        interval = FrequencyIntervalControls(
            0, 0, self.band.pencil_error_bound, self.band.max_contraction
        )
        checked = ReductionIntervalControls(
            interval, self.additional_reduced_error_bound
        )
        object.__setattr__(
            self,
            "additional_reduced_error_bound",
            checked.additional_reduced_error_bound,
        )
        target = float(
            finite_array(self.maximum_absolute_error, (), "maximum absolute error")
        )
        if target <= 0:
            raise ValueError("maximum absolute error must be positive")
        object.__setattr__(self, "maximum_absolute_error", target)


@dataclass(frozen=True)
class ReductionBandCell:
    """Closed cell contained in its paired assessment's symmetric interval."""

    lower_rad_s: float
    upper_rad_s: float
    assessment: ReductionIntervalAssessment


@dataclass(frozen=True)
class ReductionBandAssessment:
    """Complete ordered cover meeting the prescribed absolute transfer-error target.

    This is conditional floating numerical evidence about supplied finite
    models, not certified arithmetic, continuum convergence, stability or a
    measured club's acoustic/impact validity band. No relative or phase-error
    claim follows from an absolute error without a nonzero response floor.
    """

    controls: ReductionBandControls
    cells: tuple[ReductionBandCell, ...]
    evaluation_count: int

    @property
    def maximum_absolute_error_bound(self) -> float:
        return max(cell.assessment.absolute_error_bound for cell in self.cells)

    @property
    def evidence_status(self) -> str:
        return "conditional-numerical"

    @property
    def stability_status(self) -> str:
        return "unqualified"


def assess_reduction_band(
    reduction: GalerkinReduction,
    ports: DisplacementPorts,
    controls: ReductionBandControls,
) -> ReductionBandAssessment:
    """Require every closed-band cell to meet its paired inverse and error bounds."""
    if not isinstance(controls, ReductionBandControls):
        raise TypeError("expected ReductionBandControls")

    def accepted(interval: FrequencyIntervalControls) -> ReductionIntervalAssessment:
        paired = ReductionIntervalControls(
            interval, controls.additional_reduced_error_bound
        )
        result = assess_reduction_interval(reduction, ports, paired)
        if result.absolute_error_bound > controls.maximum_absolute_error:
            raise _UnresolvedIntervalError(
                "reduction transfer error target is unresolved"
            )
        return result

    cells, evaluations = _assess_cover(controls.band, accepted)
    return ReductionBandAssessment(
        controls, tuple(ReductionBandCell(*cell) for cell in cells), evaluations
    )


__all__ = ()
