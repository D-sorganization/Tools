"""Continuous coverage, bounded work and independent frequency-band controls."""

from dataclasses import replace
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest

from shared.python.golf_club import _shaft_frequency_band as band
from shared.python.golf_club._shaft_damped_spectrum import DampedPencil
from shared.python.golf_club._shaft_frequency_interval import (
    FrequencyIntervalAssessment,
    FrequencyIntervalControls,
)
from shared.python.golf_club._shaft_gripped_operating import constant_gripped_model
from shared.python.golf_club._shaft_spectrum import SpectrumScales

from .test_shaft_bending_continuum import _bending_model
from .test_shaft_equilibrium import _controls as _equilibrium_controls
from .test_shaft_frequency_interval import _oscillator
from .test_shaft_galerkin_rod import _transfer
from .test_shaft_gripped_operating import _scales


def _controls() -> band.FrequencyBandControls:
    return band.FrequencyBandControls((0, 4), 0, 0.6, 511)


def test_scalar_band_covers_the_resonance_without_gaps() -> None:
    result = band.assess_frequency_band(_oscillator(), _scales(), _controls())
    assert len(result.cells) > 1
    assert result.cells[0].lower_rad_s == 0
    assert result.cells[-1].upper_rad_s == 4
    for left, right in pairwise(result.cells):
        assert left.upper_rad_s == right.lower_rad_s
    for cell in result.cells:
        response = cell.assessment
        center = Fraction(response.controls.center_rad_s)
        width = Fraction(response.controls.half_width_rad_s)
        assert center - width <= Fraction(cell.lower_rad_s)
        assert Fraction(cell.upper_rad_s) <= center + width
        values = np.linspace(cell.lower_rad_s, cell.upper_rad_s, 13)
        exact = 1 / (4 - values**2 + 0.2j * values)
        assert np.max(np.abs(exact)) <= response.inverse_norm_bound
        assert np.max(np.abs(exact - response.inverse_array()[0, 0])) <= (
            response.inverse_difference_bound
        )
    # Independent maximum of |1/(4-w^2+0.2iw)| on this interval.
    assert result.maximum_inverse_norm_bound >= 1 / np.sqrt(0.1596)
    assert result.evaluation_count <= 511
    assert result.evidence_status == "conditional-numerical"
    assert result.stability_status == "unqualified"


def test_finite_endpoints_do_not_qualify_a_band_containing_an_undamped_pole() -> None:
    controls = replace(_controls(), bounds_rad_s=(1.3, 2.3), max_evaluations=31)
    assert np.isfinite(1 / (4 - np.array(controls.bounds_rad_s) ** 2)).all()
    with pytest.raises(ValueError, match="frequency"):
        band.assess_frequency_band(
            replace(_oscillator(), damping=[[0]]), _scales(), controls
        )


def test_budget_refusal_never_returns_only_successful_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[FrequencyIntervalControls] = []
    original = band.assess_frequency_interval

    def counted(
        pencil: DampedPencil,
        scales: SpectrumScales,
        controls: FrequencyIntervalControls,
    ) -> FrequencyIntervalAssessment:
        calls.append(controls)
        return original(pencil, scales, controls)

    monkeypatch.setattr(band, "assess_frequency_interval", counted)
    with pytest.raises(ValueError, match="evaluation budget"):
        band.assess_frequency_band(
            _oscillator(), _scales(), replace(_controls(), max_evaluations=3)
        )
    # The third attempt accepts [0,1], but the rest of [0,4] is still pending.
    assert len(calls) == 3
    assert original(_oscillator(), _scales(), calls[-1]).contraction < 0.6


def test_interval_uncertainty_cannot_be_escaped_by_subdivision() -> None:
    controls = replace(
        _controls(), bounds_rad_s=(0.5, 1), pencil_error_bound=4, max_evaluations=15
    )
    with pytest.raises(ValueError, match="evaluation budget"):
        band.assess_frequency_band(_oscillator(), _scales(), controls)


def test_adjacent_float_endpoints_refuse_unresolvable_refinement() -> None:
    controls = replace(
        _controls(), bounds_rad_s=(1, np.nextafter(1.0, np.inf)), pencil_error_bound=3
    )
    with pytest.raises(ValueError, match="representable"):
        band.assess_frequency_band(_oscillator(), _scales(), controls)


@pytest.mark.parametrize(
    "bounds",
    [
        (0, np.nextafter(0.0, 1.0)),
        (0, 1),
        (0.1, 0.2),
        (1, np.nextafter(1.0, np.inf)),
        (np.nextafter(1.0, np.inf), 1.25),
        (1e308, np.nextafter(1e308, np.inf)),
    ],
)
def test_symmetric_cells_enclose_exact_binary_endpoints(
    bounds: tuple[float, float],
) -> None:
    controls = replace(_controls(), bounds_rad_s=bounds)
    interval = band._cover_interval(*controls.bounds_rad_s, controls)
    center = Fraction(interval.center_rad_s)
    width = Fraction(interval.half_width_rad_s)
    assert 0 <= center - width <= Fraction(float(bounds[0]))
    assert Fraction(float(bounds[1])) <= center + width


@pytest.mark.parametrize("budget", [0, -1, True, np.bool_(True), 1.5, "3", np.inf])
def test_budget_requires_a_positive_integer(budget: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_controls(), max_evaluations=budget)


@pytest.mark.parametrize(
    "bounds", [(-1, 4), (4, 0), (0,), (0, np.inf), (0, np.nan), (True, 4), ("0", 4)]
)
def test_bounds_reject_invalid_units_shapes_and_domains(bounds: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_controls(), bounds_rad_s=bounds)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pencil_error_bound", -1),
        ("pencil_error_bound", True),
        ("max_contraction", 0),
        ("max_contraction", 1),
    ],
)
def test_existing_interval_scalar_contracts_remain_strict(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(_controls(), **{field: value})


def test_bounds_are_owned_and_numpy_integer_budget_is_accepted() -> None:
    bounds = np.array([0.0, 4.0])
    controls = replace(_controls(), bounds_rad_s=bounds, max_evaluations=np.int64(511))
    bounds[:] = 99
    assert controls.bounds_rad_s == (0.0, 4.0)
    assert type(controls.max_evaluations) is int


def test_single_frequency_is_a_single_assessment() -> None:
    controls = replace(_controls(), bounds_rad_s=(1, 1), max_evaluations=1)
    result = band.assess_frequency_band(_oscillator(), _scales(), controls)
    assert len(result.cells) == result.evaluation_count == 1
    assert (
        result.maximum_inverse_norm_bound
        == result.cells[0].assessment.inverse_norm_bound
    )


def test_frequency_coverage_does_not_infer_time_domain_stability() -> None:
    result = band.assess_frequency_band(_oscillator(-4), _scales(), _controls())
    assert result.stability_status == "unqualified"


def test_invalid_plant_is_not_converted_into_a_refinement_problem() -> None:
    with pytest.raises(ValueError, match="mass"):
        band.assess_frequency_band(
            replace(_oscillator(), mass=[[-1]]), _scales(), _controls()
        )


def test_coupled_nonreciprocal_band_retains_gyroscopic_and_circulatory_terms() -> None:
    pencil = DampedPencil(
        [[2, 0.1], [0.1, 1]],
        [[0, 0.3], [-0.3, 0]],
        [[0.4, 0.1], [0.1, 0.2]],
        [[8, 0.8], [-0.2, 6]],
    )
    result = band.assess_frequency_band(pencil, _scales(), _controls())
    mass, gyro, damping, stiffness = pencil.arrays()
    assert len(result.cells) > 1
    for cell in result.cells:
        assessment = cell.assessment
        for frequency in np.linspace(cell.lower_rad_s, cell.upper_rad_s, 11):
            dynamic = (
                stiffness - frequency**2 * mass + 1j * frequency * (gyro + damping)
            )
            # Explicit 2x2 adjugate gives an oracle independent of the solver.
            determinant = dynamic[0, 0] * dynamic[1, 1] - dynamic[0, 1] * dynamic[1, 0]
            exact = (
                np.array(
                    [[dynamic[1, 1], -dynamic[0, 1]], [-dynamic[1, 0], dynamic[0, 0]]]
                )
                / determinant
            )
            assert np.linalg.norm(exact, 2) <= assessment.inverse_norm_bound
            assert np.linalg.norm(exact - assessment.inverse_array(), 2) <= (
                assessment.inverse_difference_bound
            )


def test_gripped_shaft_port_is_bounded_across_every_covered_cell() -> None:
    chain, poses = _bending_model(4)
    scales = _scales()
    model = constant_gripped_model(chain, poses, _equilibrium_controls(), scales)
    controls = replace(_controls(), bounds_rad_s=(7.99, 8.01))
    result = band.assess_frequency_band(model.pencil, scales, controls)
    load = np.zeros(len(model.pencil.mass))
    load[-6] = scales.length_m  # S.T times a unit SI transverse point force.
    assert result.cells[0].lower_rad_s == controls.bounds_rad_s[0]
    assert result.cells[-1].upper_rad_s == controls.bounds_rad_s[1]
    assert all(
        left.upper_rad_s == right.lower_rad_s for left, right in pairwise(result.cells)
    )
    for cell in result.cells:
        assessment = cell.assessment
        center = load @ assessment.inverse_array() @ load
        radius = np.linalg.norm(load) ** 2 * assessment.inverse_difference_bound
        values = np.linspace(cell.lower_rad_s, cell.upper_rad_s, 11)
        exact = _transfer(model.pencil, load, values)
        assert np.max(np.abs(exact - center)) <= radius
        assert np.max(np.abs(exact)) <= abs(center) + radius


def test_wrong_controls_type_is_rejected_explicitly() -> None:
    with pytest.raises(TypeError, match="FrequencyBandControls"):
        band.assess_frequency_band(_oscillator(), _scales(), object())
