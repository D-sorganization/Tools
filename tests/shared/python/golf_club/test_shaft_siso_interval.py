"""Independent response-floor and wrapped-phase controls for shaft reduction."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club._shaft_damped_spectrum import DampedPencil
from shared.python.golf_club._shaft_galerkin import GalerkinReduction
from shared.python.golf_club._shaft_siso_interval import (
    _minimum_segment_modulus,
    assess_siso_reduction_interval,
)
from shared.python.golf_club._shaft_transfer_ports import DisplacementPorts

from .test_shaft_reduction_interval import _controls, _model, _ports


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
@pytest.mark.parametrize(
    "center,span,minimum",
    [
        (3 + 4j, 0j, 5.0),
        (3 + 4j, 2 + 0j, np.sqrt(17)),
        (1 + 2j, 3 + 4j, 0.4),
        (1 + 0j, 2 + 0j, 0.0),
        (0j, 1 + 2j, 0.0),
    ],
)
def test_complex_line_minimum_matches_independent_plane_geometry(
    scale: float,
    center: complex,
    span: complex,
    minimum: float,
) -> None:
    result = _minimum_segment_modulus(scale * center, scale * span)
    assert result == pytest.approx(scale * minimum, rel=2e-14, abs=0)


def test_full_response_floor_bounds_relative_magnitude_and_phase_between_samples() -> (
    None
):
    result = assess_siso_reduction_interval(_model(), _ports(), _controls(0.5, 0.01))
    assert 0 < result.full_response_lower_bound
    assert 0 < result.relative_complex_error_bound < 0.02
    assert result.relative_magnitude_error_bound == result.relative_complex_error_bound
    for frequency in np.linspace(0.49, 0.51, 51):
        reduced = 1 / (1 - frequency**2 + 0.2j * frequency)
        full = reduced + 1 / (100 - frequency**2 + 0.2j * frequency)
        assert abs(full) >= result.full_response_lower_bound
        assert abs(reduced - full) / abs(full) <= result.relative_complex_error_bound
        assert (
            abs(abs(reduced) / abs(full) - 1) <= result.relative_magnitude_error_bound
        )
        assert abs(np.angle(reduced / full)) <= result.phase_error_bound_rad
    assert result.evidence_status == "conditional-numerical"
    assert result.stability_status == "unqualified"


def test_phase_difference_uses_complex_ratio_across_principal_angle_wrap() -> None:
    original = _model()
    plant = replace(original.full_pencil, damping=np.diag([0.001, 0.2]))
    model = GalerkinReduction(plant, [[1], [0]], original.scales)
    ports = DisplacementPorts([[1], [1]], [[1, -1]], [1], [1])
    result = assess_siso_reduction_interval(model, ports, _controls(3, 0.001))
    reduced = 1 / complex(-8, 0.003)
    full = reduced - 1 / complex(91, 0.6)
    assert abs(np.angle(full) - np.angle(reduced)) > 6
    assert abs(np.angle(reduced / full)) < result.phase_error_bound_rad < 0.1


@pytest.mark.parametrize("width", [0.0, 0.01])
def test_exact_antiresonance_refuses_relative_and_phase_claims(width: float) -> None:
    model = _model()
    plant = DampedPencil(np.eye(2), np.zeros((2, 2)), np.zeros((2, 2)), np.diag([1, 2]))
    model = GalerkinReduction(plant, [[1], [0]], model.scales)
    ports = DisplacementPorts([[1], [1]], [[1, -2]], [1], [1])
    # H(0)=1-2/2=0 exactly; both state pencils remain invertible.
    controls = _controls(0, 0) if width == 0 else _controls(width, width)
    with pytest.raises(ValueError, match="response.*floor"):
        assess_siso_reduction_interval(model, ports, controls)


def test_zero_response_and_mimo_ports_are_not_assigned_a_phase() -> None:
    for ports in (
        DisplacementPorts([[0], [0]], [[1, 1]], [1], [1]),
        DisplacementPorts(np.eye(2), np.eye(2), [1, 1], [1, 1]),
    ):
        with pytest.raises(ValueError, match="floor|single"):
            assess_siso_reduction_interval(_model(), ports, _controls(0.5, 0.01))


def test_tiny_response_keeps_the_same_dimensionless_relative_error() -> None:
    ordinary = assess_siso_reduction_interval(_model(), _ports(), _controls(0.5, 0.01))
    tiny = DisplacementPorts([[1e-200], [1e-200]], [[1, 1]], [1], [1])
    result = assess_siso_reduction_interval(_model(), tiny, _controls(0.5, 0.01))
    assert result.full_response_lower_bound == pytest.approx(
        1e-200 * ordinary.full_response_lower_bound, rel=1e-12, abs=0
    )
    assert result.relative_complex_error_bound == pytest.approx(
        ordinary.relative_complex_error_bound, rel=1e-11
    )


def test_overestimated_center_inverse_cannot_inflate_the_response_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.linalg.solve

    def inflated(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        factor = 1.1 if np.iscomplexobj(matrix) else 1
        return factor * original(matrix, rhs)

    model = _model()
    monkeypatch.setattr(np.linalg, "solve", inflated)
    result = assess_siso_reduction_interval(model, _ports(), _controls(0, 0))
    assert result.reduction.full.center_residual_norm > 0.1
    assert 0 < result.full_response_lower_bound <= 1.01


def test_assumed_coefficient_errors_are_included_in_floor_and_ratio() -> None:
    controls = _controls(0.5, 0.01)
    controls = replace(
        controls,
        interval=replace(controls.interval, pencil_error_bound=0.001),
        additional_reduced_error_bound=0.0001,
    )
    result = assess_siso_reduction_interval(_model(), _ports(), controls)
    for frequency in np.linspace(0.49, 0.51, 11):
        for error in (0.001, -0.001, 0.001j):
            first = 1 - frequency**2 + 0.2j * frequency + error
            full = 1 / first + 1 / (100 - frequency**2 + 0.2j * frequency + error)
            for extra in (0.0001, -0.0001, 0.0001j):
                reduced = 1 / (first + extra)
                assert abs(full) >= result.full_response_lower_bound
                assert abs(reduced / full - 1) <= result.relative_complex_error_bound
                assert abs(np.angle(reduced / full)) <= result.phase_error_bound_rad


def test_undetermined_phase_and_wrong_types_are_refused() -> None:
    with pytest.raises(ValueError, match="relative|phase"):
        assess_siso_reduction_interval(_model(), _ports(), _controls(10, 0.001))
    with pytest.raises(TypeError):
        assess_siso_reduction_interval(_model(), None, _controls())


def test_segment_geometry_refuses_unrepresentable_relative_span() -> None:
    with pytest.raises(FloatingPointError):
        _minimum_segment_modulus(1e200 + 0j, 1e-200 + 0j)


def test_coupled_gyroscopic_circulatory_phase_and_floor_match_adjugate() -> None:
    original = _model()
    plant = DampedPencil(
        [[2, 0], [0, 1]],
        [[0, 0.3], [-0.3, 0]],
        [[0.4, 0.1], [0.1, 0.2]],
        [[8, 0.8], [-0.2, 6]],
    )
    model = GalerkinReduction(plant, [[1], [0]], original.scales)
    result = assess_siso_reduction_interval(model, _ports(), _controls(0.8, 0.01))
    for frequency in np.linspace(0.79, 0.81, 41):
        first = 8 - 2 * frequency**2 + 0.4j * frequency
        second = 6 - frequency**2 + 0.2j * frequency
        upper, lower = 0.8 + 0.4j * frequency, -0.2 - 0.2j * frequency
        full = (first + second - upper - lower) / (first * second - upper * lower)
        reduced = 1 / first
        assert abs(full) >= result.full_response_lower_bound
        assert abs(reduced / full - 1) <= result.relative_complex_error_bound
        assert abs(np.angle(reduced / full)) <= result.phase_error_bound_rad
