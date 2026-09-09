"""Whole-band response floors, relative magnitudes and wrapped phase errors."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from shared.python.golf_club import _shaft_siso_band as band
from shared.python.golf_club._shaft_frequency_band import FrequencyBandControls
from shared.python.golf_club._shaft_galerkin import GalerkinReduction
from shared.python.golf_club._shaft_reduction_band import ReductionBandControls
from shared.python.golf_club._shaft_transfer_ports import DisplacementPorts

from .test_shaft_reduction_interval import _model, _ports


def _controls() -> band.SisoReductionBandControls:
    absolute = ReductionBandControls(
        FrequencyBandControls((0, 0.8), 0, 0.6, 511), 0, 0.02
    )
    return band.SisoReductionBandControls(absolute, 0.02, 0.02)


def test_complete_closed_band_meets_all_three_error_targets() -> None:
    result = band.assess_siso_reduction_band(_model(), _ports(), _controls())
    assert result.cells[0].lower_rad_s == 0
    assert result.cells[-1].upper_rad_s == 0.8
    assert result.evaluation_count == 2 * len(result.cells) - 1
    assert result.minimum_full_response_lower_bound > 0
    assert result.maximum_relative_complex_error_bound <= 0.02
    assert result.maximum_relative_magnitude_error_bound <= 0.02
    assert result.maximum_phase_error_bound_rad <= 0.02
    for left, right in pairwise(result.cells):
        assert left.upper_rad_s == right.lower_rad_s
    for cell in result.cells:
        frequencies = np.linspace(cell.lower_rad_s, cell.upper_rad_s, 11)
        reduced = 1 / (1 - frequencies**2 + 0.2j * frequencies)
        full = reduced + 1 / (100 - frequencies**2 + 0.2j * frequencies)
        checked = cell.assessment
        assert np.min(np.abs(full)) >= checked.full_response_lower_bound
        assert (
            np.max(np.abs(full - reduced))
            <= checked.reduction.absolute_error_bound
            <= 0.02
        )
        assert (
            np.max(np.abs(reduced / full - 1)) <= checked.relative_complex_error_bound
        )
        assert np.max(np.abs(np.angle(reduced / full))) <= checked.phase_error_bound_rad
    assert result.evidence_status == "conditional-numerical"
    assert result.stability_status == "unqualified"


def test_unmet_relative_error_is_not_hidden_by_small_cells() -> None:
    controls = _controls()
    absolute = replace(
        controls.absolute, band=replace(controls.absolute.band, max_evaluations=31)
    )
    controls = replace(controls, absolute=absolute, maximum_relative_error=1e-4)
    with pytest.raises(ValueError, match="budget|representable"):
        band.assess_siso_reduction_band(_model(), _ports(), controls)


def test_band_with_a_transfer_zero_cannot_report_relative_or_phase_agreement() -> None:
    model = _model()
    plant = replace(
        model.full_pencil, stiffness=np.diag([1, 2]), damping=np.zeros((2, 2))
    )
    model = GalerkinReduction(plant, np.eye(2), model.scales)
    ports = DisplacementPorts([[1], [1]], [[1, -2]], [1], [1])
    absolute = ReductionBandControls(
        FrequencyBandControls((0, 0.01), 0, 0.6, 31), 0, 0.02
    )
    controls = band.SisoReductionBandControls(absolute, 0.02, 0.02)
    with pytest.raises(ValueError, match="budget|representable"):
        band.assess_siso_reduction_band(model, ports, controls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("maximum_relative_error", 0),
        ("maximum_relative_error", 1),
        ("maximum_relative_error", True),
        ("maximum_relative_error", "0.1"),
        ("maximum_relative_error", np.nan),
        ("maximum_relative_error", 1j),
        ("maximum_phase_error_rad", 0),
        ("maximum_phase_error_rad", np.pi / 2),
        ("maximum_phase_error_rad", np.inf),
        ("maximum_phase_error_rad", False),
    ],
)
def test_relative_and_phase_targets_have_strict_domains(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_controls(), **{field: value})


def test_wrong_controls_and_ports_are_refused() -> None:
    with pytest.raises(TypeError):
        replace(_controls(), absolute=None)
    with pytest.raises(TypeError):
        band.assess_siso_reduction_band(_model(), _ports(), None)
    with pytest.raises(ValueError, match="single"):
        band.assess_siso_reduction_band(
            _model(),
            DisplacementPorts(np.eye(2), np.eye(2), [1, 1], [1, 1]),
            _controls(),
        )
