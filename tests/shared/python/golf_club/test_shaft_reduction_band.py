"""Complete-band transfer-error acceptance, without sparse-sample substitutes."""

from dataclasses import replace
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest

from shared.python.golf_club import _shaft_reduction_band as band
from shared.python.golf_club._shaft_frequency_band import FrequencyBandControls
from shared.python.golf_club._shaft_frequency_interval import FrequencyIntervalControls
from shared.python.golf_club._shaft_galerkin import GalerkinReduction
from shared.python.golf_club._shaft_reduction_interval import (
    ReductionIntervalAssessment,
    ReductionIntervalControls,
)
from shared.python.golf_club._shaft_transfer_ports import DisplacementPorts

from .test_shaft_reduction_interval import _model, _ports


def _controls() -> band.ReductionBandControls:
    return band.ReductionBandControls(
        FrequencyBandControls((9, 11), 0, 0.6, 2047), 0, 0.6
    )


def test_entire_band_meets_absolute_error_limit_and_contains_true_peak() -> None:
    result = band.assess_reduction_band(_model(), _ports(), _controls())
    assert result.cells[0].lower_rad_s == 9
    assert result.cells[-1].upper_rad_s == 11
    assert result.evaluation_count == 2 * len(result.cells) - 1
    for left, right in pairwise(result.cells):
        assert left.upper_rad_s == right.lower_rad_s
    for cell in result.cells:
        interval = cell.assessment.controls.interval
        center, width = (
            Fraction(interval.center_rad_s),
            Fraction(interval.half_width_rad_s),
        )
        assert center - width <= Fraction(cell.lower_rad_s)
        assert center + width >= Fraction(cell.upper_rad_s)
        frequencies = np.linspace(cell.lower_rad_s, cell.upper_rad_s, 7)
        exact_error = np.abs(1 / (100 - frequencies**2 + 0.2j * frequencies))
        assert np.max(exact_error) <= cell.assessment.absolute_error_bound <= 0.6
    assert (
        1 / np.sqrt(0.2**2 * 100 - 0.2**4 / 4)
        <= result.maximum_absolute_error_bound
        <= 0.6
    )
    assert result.evidence_status == "conditional-numerical"
    assert result.stability_status == "unqualified"


def test_too_few_modes_cannot_meet_error_target_by_sampling_around_peak() -> None:
    # Endpoint omitted responses meet 0.1, but the interior peak exceeds 0.5.
    endpoints = np.array([9, 11])
    assert np.max(np.abs(1 / (100 - endpoints**2 + 0.2j * endpoints))) < 0.1
    controls = replace(
        _controls(),
        maximum_absolute_error=0.1,
        band=replace(_controls().band, max_evaluations=63),
    )
    with pytest.raises(ValueError, match="budget|representable"):
        band.assess_reduction_band(_model(), _ports(), controls)


def test_single_frequency_full_basis_can_meet_roundoff_scale_target() -> None:
    controls = replace(
        _controls(),
        band=FrequencyBandControls((3, 3), 0, 0.6, 1),
        maximum_absolute_error=1e-12,
    )
    result = band.assess_reduction_band(
        _model([[2, 0.3], [-0.2, 0.7]]), _ports(), controls
    )
    assert result.evaluation_count == len(result.cells) == 1
    assert result.maximum_absolute_error_bound < 1e-12


def test_budget_counts_rejected_pairs_and_refuses_partial_cover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ReductionIntervalAssessment | None] = []
    original = band.assess_reduction_interval

    def counted(
        model: GalerkinReduction,
        ports: DisplacementPorts,
        controls: ReductionIntervalControls,
    ) -> ReductionIntervalAssessment:
        calls.append(None)
        result = original(model, ports, controls)
        calls[-1] = result
        return result

    monkeypatch.setattr(band, "assess_reduction_interval", counted)
    controls = replace(_controls(), band=replace(_controls().band, max_evaluations=15))
    with pytest.raises(ValueError, match="evaluation budget"):
        band.assess_reduction_band(_model(), _ports(), controls)
    assert len(calls) == 15
    assert any(
        result is not None and result.absolute_error_bound < 0.6 for result in calls
    )


@pytest.mark.parametrize("value", [0, -1, True, "1", np.inf, np.nan, 1j])
def test_error_acceptance_target_is_positive_finite_and_strict(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_controls(), maximum_absolute_error=value)


def test_wrong_band_control_type_is_refused() -> None:
    with pytest.raises(TypeError):
        replace(_controls(), band=FrequencyIntervalControls(1, 0, 0, 0.6))
    with pytest.raises(TypeError):
        band.assess_reduction_band(_model(), _ports(), None)
