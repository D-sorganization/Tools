"""Independent transfer controls for errors caused by omitted shaft modes."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club._shaft_damped_spectrum import DampedPencil
from shared.python.golf_club._shaft_frequency_interval import FrequencyIntervalControls
from shared.python.golf_club._shaft_galerkin import GalerkinReduction
from shared.python.golf_club._shaft_reduction_interval import (
    ReductionIntervalControls,
    assess_reduction_interval,
)
from shared.python.golf_club._shaft_transfer_ports import DisplacementPorts

from .test_shaft_affine_transient import _scales


def _model(basis: object = ((1,), (0,))) -> GalerkinReduction:
    plant = DampedPencil(
        np.eye(2), np.zeros((2, 2)), 0.2 * np.eye(2), np.diag([1, 100])
    )
    return GalerkinReduction(plant, basis, _scales())


def _ports() -> DisplacementPorts:
    return DisplacementPorts([[1], [1]], [[1, 1]], [1], [1])


def _controls(center: float = 10, width: float = 0.001) -> ReductionIntervalControls:
    return ReductionIntervalControls(
        FrequencyIntervalControls(center, width, 0, 0.9), 0
    )


def test_omitted_resonance_peak_is_bounded_throughout_closed_interval() -> None:
    # The scalar omitted mode peaks at sqrt(k-c^2/2), not exactly sqrt(k).
    center = np.sqrt(100 - 0.2**2 / 2)
    result = assess_reduction_interval(_model(), _ports(), _controls(center, 0.001))
    peak = 1 / np.sqrt(0.2**2 * 100 - 0.2**4 / 4)
    exact_center = 1 / (100 - center**2 + 0.2j * center)
    assert result.center_error_array()[0, 0] == pytest.approx(exact_center)
    assert peak <= result.absolute_error_bound < 1.1 * peak
    for frequency in np.linspace(center - 0.001, center + 0.001, 101):
        omitted = 1 / (100 - frequency**2 + 0.2j * frequency)
        assert abs(omitted) <= result.absolute_error_bound
    assert result.evidence_status == "conditional-numerical"
    assert result.stability_status == "unqualified"


def test_full_nonorthogonal_basis_and_channel_units_preserve_center_transfer() -> None:
    model = _model([[2, 0.3], [-0.2, 0.7]])
    result = assess_reduction_interval(model, _ports(), _controls(3, 0))
    assert np.max(np.abs(result.center_error_array())) < 1e-15
    assert result.absolute_error_bound < 1e-13
    # Changing numerical units of each physical channel and its scale cancels.
    equivalent = DisplacementPorts([[0.001], [0.001]], [[1000, 1000]], [1000], [1000])
    changed = assess_reduction_interval(model, equivalent, _controls(3, 0))
    np.testing.assert_array_equal(
        changed.center_error_array(), result.center_error_array()
    )
    assert changed.absolute_error_bound == result.absolute_error_bound


def test_nonsymmetric_gyro_circulatory_response_matches_independent_adjugate() -> None:
    plant = DampedPencil(
        [[2, 0], [0, 1]],
        [[0, 0.3], [-0.3, 0]],
        [[0.4, 0.1], [0.1, 0.2]],
        [[8, 0.8], [-0.2, 6]],
    )
    model = GalerkinReduction(plant, [[1], [0]], _scales())
    result = assess_reduction_interval(model, _ports(), _controls(0.8, 0.01))
    for frequency in np.linspace(0.79, 0.81, 41):
        first = 8 - 2 * frequency**2 + 0.4j * frequency
        second = 6 - frequency**2 + 0.2j * frequency
        upper = 0.8 + 0.4j * frequency
        lower = -0.2 - 0.2j * frequency
        exact = (first + second - upper - lower) / (first * second - upper * lower)
        error = exact - 1 / first
        assert abs(error) <= result.absolute_error_bound
        if frequency == 0.8:
            assert result.center_error_array()[0, 0] == pytest.approx(error)


def test_full_uncertainty_is_projected_and_extra_reduced_error_is_explicit() -> None:
    model = _model([[2], [0]])
    controls = ReductionIntervalControls(
        FrequencyIntervalControls(3, 0.01, 0.03, 0.9), 0.02
    )
    result = assess_reduction_interval(model, _ports(), controls)
    assert result.reduced.controls.pencil_error_bound == pytest.approx(4 * 0.03 + 0.02)
    for frequency in np.linspace(2.99, 3.01, 11):
        for error in (0.03, -0.03, 0.03j):
            for additional in (0.02, -0.02, 0.02j):
                first = 1 - frequency**2 + 0.2j * frequency
                second = 100 - frequency**2 + 0.2j * frequency
                exact = 1 / (first + error) + 1 / (second + error)
                reduced = 4 / (4 * (first + error) + additional)
                assert abs(exact - reduced) <= result.absolute_error_bound


def test_absolute_error_survives_full_response_antiresonance() -> None:
    plant = replace(_model().full_pencil, damping=np.zeros((2, 2)))
    model = GalerkinReduction(plant, [[1], [0]], _scales())
    frequency = np.sqrt(50.5)
    result = assess_reduction_interval(model, _ports(), _controls(frequency, 0.001))
    assert abs(1 / (1 - frequency**2) + 1 / (100 - frequency**2)) < 1e-15
    assert abs(1 / (100 - frequency**2)) <= result.absolute_error_bound


def test_hidden_full_pole_cannot_be_erased_by_projection_or_zero_ports() -> None:
    plant = replace(_model().full_pencil, damping=np.zeros((2, 2)))
    model = GalerkinReduction(plant, [[1], [0]], _scales())
    for ports in (_ports(), DisplacementPorts([[0], [0]], [[0, 0]], [1], [1])):
        with pytest.raises(ValueError):
            assess_reduction_interval(model, ports, _controls(9.9, 0.2))


def test_imperfect_center_solves_retain_their_residual_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.linalg.solve

    def biased(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        factor = 0.9 if np.iscomplexobj(matrix) and len(matrix) == 2 else 1
        return factor * original(matrix, rhs)

    model = _model(np.eye(2))
    monkeypatch.setattr(np.linalg, "solve", biased)
    result = assess_reduction_interval(model, _ports(), _controls(3, 0))
    assert result.full.center_residual_norm > 0.1
    assert result.absolute_error_bound > 0.01


@pytest.mark.parametrize("extra", [-1, float("inf"), True, "0", 1j])
def test_invalid_additional_projection_error_is_refused(extra: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ReductionIntervalControls(FrequencyIntervalControls(1, 0, 0, 0.9), extra)


def test_wrong_types_and_port_topology_are_refused() -> None:
    with pytest.raises(TypeError):
        ReductionIntervalControls(None, 0)
    for arguments in (
        (None, _ports(), _controls()),
        (_model(), None, _controls()),
        (_model(), _ports(), None),
    ):
        with pytest.raises(TypeError):
            assess_reduction_interval(*arguments)
    with pytest.raises(ValueError):
        assess_reduction_interval(
            _model(), DisplacementPorts([[1]], [[1]], [1], [1]), _controls()
        )


def test_center_error_result_is_owned() -> None:
    result = assess_reduction_interval(_model(), _ports(), _controls())
    expected = result.center_error_array().copy()
    result.center_error_array()[:] = 0
    np.testing.assert_array_equal(result.center_error_array(), expected)


def test_multiple_normalized_inputs_outputs_bound_rank_one_omitted_transfer() -> None:
    ports = DisplacementPorts([[1, 2], [3, 4]], [[1, 2], [-1, 3]], [2, 0.5], [0.1, 2])
    result = assess_reduction_interval(_model(), ports, _controls())
    # Independent outer product of normalized mode-two observation and loading.
    omitted_map = np.outer([20, 1.5], [6, 2])
    np.testing.assert_allclose(
        result.center_error_array(), omitted_map / (2j), atol=1e-13
    )
    for frequency in np.linspace(9.999, 10.001, 31):
        exact_norm = (
            np.linalg.norm([20, 1.5])
            * np.linalg.norm([6, 2])
            / abs(100 - frequency**2 + 0.2j * frequency)
        )
        assert exact_norm <= result.absolute_error_bound
    assert result.absolute_error_bound == min(
        result.inverse_variation_bound, result.residual_polynomial_bound
    )


def test_tiny_positive_transfer_cannot_silently_underflow_to_zero() -> None:
    ports = DisplacementPorts([[1e-200], [1e-200]], [[1e-200, 1e-200]], [1], [1])
    with pytest.raises(ValueError, match="numerical"):
        assess_reduction_interval(_model(), ports, _controls(3, 0))
