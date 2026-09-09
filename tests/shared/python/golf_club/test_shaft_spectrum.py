"""Independent oscillator, gyroscopic and defective-mode spectral controls."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eigh

from shared.python.golf_club._rotating_body_contracts import RotatingFrameState
from shared.python.golf_club._shaft_equilibrium import solve_clamped_chain
from shared.python.golf_club._shaft_loaded_dynamics import linearized_chain_dynamics
from shared.python.golf_club._shaft_spectrum import (
    SpectrumScales,
    _frozen_spectrum,
    clamped_chain_spectrum,
)

from .test_shaft_equilibrium import _controls
from .test_shaft_rotating_chain import _radial_rod


def _scales() -> SpectrumScales:
    return SpectrumScales(1.0, 0.02, 1e-12, 1e-9)


def _assert_roots(actual: np.ndarray, expected: np.ndarray) -> None:
    # Compare unordered multisets without presuming a repeated-mode basis.
    remaining = list(actual)
    for root in expected:
        index = int(np.argmin(np.abs(np.array(remaining) - root)))
        assert remaining.pop(index) == pytest.approx(root, abs=2e-9, rel=2e-9)
    assert not remaining


def test_uncoupled_oscillators_recover_physical_rates_and_mode_equations() -> None:
    mass, frequencies = np.diag([2.0, 3.0]), np.array([4.0, 9.0])
    stiffness = mass @ np.diag(frequencies**2)
    result = _frozen_spectrum(mass, np.zeros((2, 2)), stiffness, _scales())
    _assert_roots(result.rates_s_inv, np.r_[1j * frequencies, -1j * frequencies])
    np.testing.assert_allclose(
        result.velocity_modes,
        result.displacement_modes * result.rates_s_inv,
        atol=2e-13,
    )
    assert np.max(result.relative_residuals) < 1e-13
    assert np.max(result.polynomial_relative_residuals) < 1e-13
    assert result.scaled_eigenbasis_rcond > 0
    assert result.stability_status == "unqualified"


def test_negative_stiffness_growth_and_gyroscopic_stabilization_are_both_retained() -> (
    None
):
    mass, stiffness = np.eye(2), -np.eye(2)
    no_spin = _frozen_spectrum(mass, np.zeros((2, 2)), stiffness, _scales())
    _assert_roots(no_spin.rates_s_inv, np.array([1, 1, -1, -1]))
    # det(s² I + 3 s J - I) = s⁴ + 7 s² + 1.
    gyro = np.array([[0, -3.0], [3.0, 0]])
    spinning = _frozen_spectrum(mass, gyro, stiffness, _scales())
    frequencies = np.array([(3 + np.sqrt(5)) / 2, (3 - np.sqrt(5)) / 2])
    _assert_roots(spinning.rates_s_inv, np.r_[1j * frequencies, -1j * frequencies])
    assert spinning.scaled_eigenbasis_rcond > 1e-3
    assert spinning.stability_status == "unqualified"


def test_zero_real_parts_do_not_hide_a_defective_free_particle() -> None:
    result = _frozen_spectrum(np.eye(1), np.zeros((1, 1)), np.zeros((1, 1)), _scales())
    np.testing.assert_array_equal(result.rates_s_inv, 0)
    assert result.scaled_eigenbasis_rcond < 1e-12
    # q(t)=q(0)+v(0)t is unbounded despite every eigenvalue being zero.
    assert result.stability_status == "unqualified"


def test_circulatory_stiffness_is_not_projected_to_a_stable_symmetric_matrix() -> None:
    stiffness = np.array([[4.0, 1.0], [-1.0, 4.0]])
    result = _frozen_spectrum(np.eye(2), np.zeros((2, 2)), stiffness, _scales())
    roots = np.sqrt(np.array([-4 + 1j, -4 - 1j]))
    _assert_roots(result.rates_s_inv, np.r_[roots, -roots])
    assert np.max(result.rates_s_inv.real) > 0.2


def test_time_scaling_changes_conditioning_but_not_physical_rates() -> None:
    mass, gyro, stiffness = (
        np.diag([2.0, 3.0]),
        np.zeros((2, 2)),
        np.diag([32.0, 243.0]),
    )
    first = _frozen_spectrum(mass, gyro, stiffness, _scales())
    second = _frozen_spectrum(mass, gyro, stiffness, replace(_scales(), time_s=0.3))
    _assert_roots(first.rates_s_inv, second.rates_s_inv)
    assert first.scaled_eigenbasis_rcond != pytest.approx(
        second.scaled_eigenbasis_rcond
    )


@pytest.mark.parametrize(
    "mass", [np.diag([1, 0]), np.diag([1, -1]), np.diag([1, 1e-15])]
)
def test_singular_indefinite_or_unresolved_mass_is_refused(mass: np.ndarray) -> None:
    with pytest.raises(ValueError, match="mass"):
        _frozen_spectrum(mass, np.zeros((2, 2)), np.eye(2), _scales())


def test_unloaded_chain_matches_independent_symmetric_generalized_modes() -> None:
    frame = RotatingFrameState("observer", (0, 0, 0), (0, 0, 0), (0, 0, 0))
    chain, poses = _radial_rod(2, frame)
    operators = linearized_chain_dynamics(chain, poses)
    frequencies = np.sqrt(
        eigh(operators.stiffness[6:, 6:], operators.mass[6:, 6:], eigvals_only=True)
    )
    first = clamped_chain_spectrum(chain, poses, _controls(), _scales())
    _assert_roots(first.rates_s_inv, np.r_[1j * frequencies, -1j * frequencies])
    second = clamped_chain_spectrum(
        chain, poses, _controls(), replace(_scales(), length_m=0.1)
    )
    _assert_roots(first.rates_s_inv, second.rates_s_inv)
    assert first.displacement_modes.shape == (12, 24)


def test_rotating_chain_rechecks_balance_and_material_domain() -> None:
    frame = RotatingFrameState("observer", (20, 0, 0), (0, 0, 0), (0, 0, 0))
    chain, seed = _radial_rod(2, frame)
    with pytest.raises(ValueError, match="balance"):
        clamped_chain_spectrum(chain, seed, _controls(), _scales())
    root = solve_clamped_chain(chain, seed, _controls())
    result = clamped_chain_spectrum(chain, root.poses, _controls(), _scales())
    assert np.max(result.relative_residuals) < _scales().residual_tolerance
    np.testing.assert_allclose(result.support_wrench, root.support_wrench, atol=1e-12)
    with pytest.raises(ValueError, match="strain"):
        clamped_chain_spectrum(
            chain,
            root.poses,
            replace(_controls(), strain_limits=(1e-4,) * 6),
            _scales(),
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("length_m", 0),
        ("time_s", np.inf),
        ("mass_rcond_floor", 1),
        ("residual_tolerance", -1),
    ],
)
def test_spectral_controls_require_explicit_finite_resolved_scales(
    name: str, value: float
) -> None:
    with pytest.raises(ValueError):
        replace(_scales(), **{name: value})


@pytest.mark.parametrize("value", [True, "1", 1 + 0j])
def test_scale_values_do_not_coerce_nonreal_or_boolean_input(value: object) -> None:
    with pytest.raises(TypeError):
        replace(_scales(), time_s=value)


def test_matrix_contracts_reject_asymmetric_mass_and_dissipative_gyro() -> None:
    with pytest.raises(ValueError, match="mass"):
        _frozen_spectrum([[1, 1], [0, 1]], np.zeros((2, 2)), np.eye(2), _scales())
    with pytest.raises(ValueError, match="gyroscopic"):
        _frozen_spectrum(np.eye(2), np.eye(2), np.eye(2), _scales())
    with pytest.raises(TypeError):
        _frozen_spectrum(
            np.eye(2, dtype=complex), np.zeros((2, 2)), np.eye(2), _scales()
        )
    with pytest.raises(ValueError):
        _frozen_spectrum(np.eye(2), np.zeros((1, 1)), np.eye(2), _scales())


def test_inaccurate_eigensolution_fails_the_residual_postcondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(np.linalg, "eig", lambda _: (np.array([1.0, -1.0]), np.eye(2)))
    with pytest.raises(ValueError, match="residual"):
        _frozen_spectrum(np.eye(1), np.zeros((1, 1)), np.eye(1), _scales())


def test_underflowing_time_scale_cannot_erase_physical_stiffness() -> None:
    with pytest.raises(ValueError, match="scale"):
        _frozen_spectrum(
            np.eye(1), np.zeros((1, 1)), np.eye(1), replace(_scales(), time_s=1e-200)
        )


def test_input_and_output_arrays_are_independent() -> None:
    mass, gyro, stiffness = np.eye(2), np.zeros((2, 2)), np.diag([4.0, 9.0])
    originals = tuple(array.copy() for array in (mass, gyro, stiffness))
    result = _frozen_spectrum(mass, gyro, stiffness, _scales())
    result.displacement_modes[:] = 0
    for actual, expected in zip((mass, gyro, stiffness), originals, strict=True):
        np.testing.assert_array_equal(actual, expected)
    fresh = _frozen_spectrum(mass, gyro, stiffness, _scales())
    assert np.linalg.norm(fresh.displacement_modes) > 0


def test_small_state_residual_cannot_hide_a_bad_high_frequency_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_eig = np.linalg.eig

    def corrupt_fast_mode(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rates, modes = original_eig(matrix)
        index = int(np.argmax(np.abs(rates)))
        rates[index] *= 2
        modes[2:, index] = rates[index] * modes[:2, index]
        # State scaling hides a bad fast mode in the much larger ||A||.
        residual = np.linalg.norm(
            matrix @ modes[:, index] - rates[index] * modes[:, index]
        )
        assert residual / np.linalg.norm(matrix) < _scales().residual_tolerance
        return rates, modes

    monkeypatch.setattr(np.linalg, "eig", corrupt_fast_mode)
    with pytest.raises(ValueError, match="residual"):
        _frozen_spectrum(np.eye(2), np.zeros((2, 2)), np.diag([1.0, 1e26]), _scales())
