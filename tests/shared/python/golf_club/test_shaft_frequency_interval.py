"""Independent inverse/oscillator controls for between-sample response bounds."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club._shaft_damped_spectrum import DampedPencil
from shared.python.golf_club._shaft_frequency_interval import (
    FrequencyIntervalControls,
    assess_frequency_interval,
)
from shared.python.golf_club._shaft_gripped_operating import constant_gripped_model

from .test_shaft_bending_continuum import _bending_model
from .test_shaft_equilibrium import _controls
from .test_shaft_galerkin_rod import _transfer
from .test_shaft_gripped_operating import _scales


def _oscillator(stiffness: float = 4.0) -> DampedPencil:
    return DampedPencil([[1.0]], [[0.0]], [[0.2]], [[stiffness]])


def test_scalar_interval_matches_independent_neumann_formula() -> None:
    controls = FrequencyIntervalControls(1.0, 0.1, 0.03, 0.9)
    result = assess_frequency_interval(_oscillator(), _scales(), controls)
    inverse = 1 / complex(3.0, 0.2)
    contraction = abs(inverse) * (0.1 * abs(complex(-2, 0.2)) + 0.01 + 0.03)
    assert result.contraction == pytest.approx(contraction, abs=1e-14)
    assert result.inverse_norm_bound == pytest.approx(abs(inverse) / (1 - contraction))
    assert result.inverse_difference_bound == pytest.approx(
        abs(inverse) * contraction / (1 - contraction)
    )
    for frequency in np.linspace(0.9, 1.1, 101):
        for error in (0.03, -0.03, 0.03j, -0.03j):
            exact = 1 / (4 - frequency**2 + 0.2j * frequency + error)
            assert abs(exact) <= result.inverse_norm_bound
            assert abs(exact - inverse) <= result.inverse_difference_bound
    assert result.evidence_status == "conditional-numerical"
    assert result.stability_status == "unqualified"


def test_interval_refuses_an_undamped_pole_hidden_between_endpoint_samples() -> None:
    pencil = replace(_oscillator(), damping=[[0.0]])
    # Endpoint responses are finite, but the pole at 2 rad/s is inside.
    assert np.isfinite(1 / (4 - np.array([1.3, 2.3]) ** 2)).all()
    with pytest.raises(ValueError, match="interval.*unresolved"):
        assess_frequency_interval(
            pencil, _scales(), FrequencyIntervalControls(1.8, 0.5, 0, 0.9)
        )


def test_nonsymmetric_gyro_and_stiffness_keep_full_complex_inverse() -> None:
    pencil = DampedPencil(
        [[2, 0.1], [0.1, 1]],
        [[0, 0.3], [-0.3, 0]],
        [[0.4, 0.1], [0.1, 0.2]],
        [[8, 0.8], [-0.2, 6]],
    )
    controls = FrequencyIntervalControls(0.8, 0.02, 0, 0.9)
    result = assess_frequency_interval(pencil, _scales(), controls)
    mass, gyro, damping, stiffness = pencil.arrays()
    center = np.linalg.inv(stiffness - 0.8**2 * mass + 0.8j * (gyro + damping))
    for frequency in np.linspace(0.78, 0.82, 21):
        exact = np.linalg.inv(
            stiffness - frequency**2 * mass + 1j * frequency * (gyro + damping)
        )
        assert np.linalg.norm(exact, 2) <= result.inverse_norm_bound
        assert np.linalg.norm(exact - center, 2) <= result.inverse_difference_bound


@pytest.mark.parametrize(
    "field,value",
    [
        ("center_rad_s", True),
        ("center_rad_s", "1"),
        ("center_rad_s", -1),
        ("half_width_rad_s", -1),
        ("half_width_rad_s", float("inf")),
        ("pencil_error_bound", -0.1),
        ("pencil_error_bound", 1j),
        ("max_contraction", 0),
        ("max_contraction", 1),
    ],
)
def test_interval_controls_reject_invalid_domain(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(FrequencyIntervalControls(1, 0.1, 0, 0.9), **{field: value})


def test_interval_rejects_negative_frequency_extent() -> None:
    with pytest.raises(ValueError, match="nonnegative frequency"):
        FrequencyIntervalControls(1, 2, 0, 0.9)


def test_uncertainty_can_refuse_an_otherwise_resolved_interval() -> None:
    controls = FrequencyIntervalControls(1, 0.1, 0, 0.9)
    assess_frequency_interval(_oscillator(), _scales(), controls)
    with pytest.raises(ValueError, match="interval.*unresolved"):
        assess_frequency_interval(
            _oscillator(), _scales(), replace(controls, pencil_error_bound=3)
        )


def test_frequency_bound_does_not_infer_stability() -> None:
    result = assess_frequency_interval(
        _oscillator(-4), _scales(), FrequencyIntervalControls(1, 0.1, 0, 0.9)
    )
    assert result.stability_status == "unqualified"


def test_zero_width_static_inverse_is_exact_for_scalar_control() -> None:
    result = assess_frequency_interval(
        _oscillator(), _scales(), FrequencyIntervalControls(0, 0, 0, 0.9)
    )
    assert result.inverse_norm_bound == 0.25
    assert result.inverse_difference_bound == 0


def test_original_invalid_mass_is_not_hidden_by_invertible_dynamic_stiffness() -> None:
    with pytest.raises(ValueError):
        assess_frequency_interval(
            replace(_oscillator(), mass=[[-1]]),
            _scales(),
            FrequencyIntervalControls(1, 0.1, 0, 0.9),
        )


def test_center_inverse_is_owned_and_exposed_as_fresh_complex_array() -> None:
    result = assess_frequency_interval(
        _oscillator(), _scales(), FrequencyIntervalControls(1, 0.1, 0, 0.9)
    )
    first = result.inverse_array()
    expected = 1 / complex(3, 0.2)
    assert first[0, 0] == pytest.approx(expected)
    first[0, 0] = 0
    assert result.inverse_array()[0, 0] == pytest.approx(expected)


def test_computed_inverse_defect_is_included_in_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.linalg.solve

    def imperfect_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        result = original(matrix, rhs)
        return 0.9 * result if np.iscomplexobj(matrix) else result

    monkeypatch.setattr(np.linalg, "solve", imperfect_solve)
    result = assess_frequency_interval(
        _oscillator(), _scales(), FrequencyIntervalControls(1, 0, 0, 0.9)
    )
    assert result.center_residual_norm == pytest.approx(0.1)
    assert result.contraction == pytest.approx(0.1)
    expected = 1 / complex(3, 0.2)
    assert result.inverse_difference_bound == pytest.approx(abs(expected) * 0.1)


@pytest.mark.parametrize("frequency,width", [(1e200, 1), (1e-200, 1e-201)])
def test_unrepresentable_frequency_arithmetic_is_refused(
    frequency: float, width: float
) -> None:
    with pytest.raises(ValueError, match="numerical"):
        assess_frequency_interval(
            _oscillator(),
            _scales(),
            FrequencyIntervalControls(frequency, width, 0, 0.9),
        )


def test_singular_center_is_not_regularized() -> None:
    with pytest.raises(ValueError, match="numerical"):
        assess_frequency_interval(
            replace(_oscillator(), damping=[[0]]),
            _scales(),
            FrequencyIntervalControls(2, 0.1, 0, 0.9),
        )


def test_existing_gripped_shaft_port_magnitude_and_phase_fit_interval_disk() -> None:
    chain, poses = _bending_model(4)
    scales = _scales()
    model = constant_gripped_model(chain, poses, _controls(), scales)
    controls = FrequencyIntervalControls(8, 0.001, 0, 0.9)
    result = assess_frequency_interval(model.pencil, scales, controls)
    load = np.zeros(len(model.pencil.mass))
    load[-6] = scales.length_m  # S.T times a unit SI transverse point force.
    center = load @ result.inverse_array() @ load
    radius = np.linalg.norm(load) ** 2 * result.inverse_difference_bound
    assert 0 < radius < abs(center)
    samples = _transfer(model.pencil, load, np.linspace(7.999, 8.001, 11))
    assert np.max(np.abs(samples - center)) <= radius
    assert np.max(np.abs(np.abs(samples) - abs(center))) <= radius
    assert np.max(np.abs(np.angle(samples / center))) <= np.arcsin(radius / abs(center))
