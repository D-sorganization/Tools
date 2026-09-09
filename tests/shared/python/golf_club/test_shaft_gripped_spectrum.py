"""Dynamic-boundary rod poles and all-node finite-grip spectrum controls."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import root

from shared.python.golf_club._shaft_gripped_equilibrium import solve_gripped_chain
from shared.python.golf_club._shaft_gripped_spectrum import gripped_chain_spectrum

from .test_shaft_equilibrium import _controls
from .test_shaft_gripped_response import _model
from .test_shaft_spectrum import _assert_roots, _scales


def _axial_rates(result: object) -> np.ndarray:
    modes = result.displacement_modes
    axial = np.linalg.norm(modes[2::6], axis=0)
    return result.rates_s_inv[axial > (1 - 1e-8) * np.linalg.norm(modes, axis=0)]


def _continuum_residual(parts: np.ndarray) -> list[float]:
    # EA*u''=mu*s^2*u, root EA*u'=dg*u, tip EA*u'+mt*s^2*u=0.
    rate = complex(*parts)
    wave = rate * np.sqrt(0.2 / 1000)
    grip = 400 + 2 * rate + 0.03 * rate**2
    shape = np.cosh(wave) + grip * np.sinh(wave) / (1000 * wave)
    residual = (
        1000 * wave * np.sinh(wave) + grip * np.cosh(wave) + 0.1 * rate**2 * shape
    )
    return [residual.real, residual.imag]


def test_all_node_axial_poles_match_independent_two_node_polynomial() -> None:
    chain, poses = _model()
    result = gripped_chain_spectrum(chain, poses, _controls(), _scales())
    # det(s²M+sC+K): assembled by scalar two-node rod balance, not model matrices.
    a, b, d = 0.03 + 0.2 / 3, 0.2 / 6, 0.1 + 0.2 / 3
    expected = np.roots(
        [a * d - b * b, 2 * d, a * 1000 + d * 1400 + 2000 * b, 2000, 400000]
    )
    _assert_roots(_axial_rates(result), expected)
    assert result.displacement_modes.shape == (12, 24)
    assert result.frame == chain.shaft.frame
    assert result.support_wrench is None
    assert result.stability_status == "unqualified"
    np.testing.assert_array_equal(result.grip_wrenches[0], 0)
    np.testing.assert_allclose(
        result.velocity_modes,
        result.displacement_modes * result.rates_s_inv,
        atol=1e-11,
    )


@pytest.mark.parametrize("seed", [[-2.2, 32.2], [-6.7, 153.6]])
def test_damped_poles_converge_to_independent_continuum_boundaries(seed: list) -> None:
    reference = root(_continuum_residual, seed, tol=1e-11)
    assert reference.success
    assert np.linalg.norm(_continuum_residual(reference.x)) < 1e-9
    exact = complex(*reference.x)
    errors = []
    for count in (2, 4, 8):
        chain, poses = _model(count)
        result = gripped_chain_spectrum(chain, poses, _controls(), _scales())
        rates = _axial_rates(result)
        observed = rates[np.argmin(abs(rates - exact))]
        errors.append(abs(observed - exact))
    assert 3.5 < errors[0] / errors[1] < 4.5
    assert 3.5 < errors[1] / errors[2] < 4.5
    assert errors[-1] / abs(exact) < 0.002


def test_coordinate_scaling_and_rotating_preload_preserve_physical_modes() -> None:
    chain, seed = _model(2, spin=10)
    equilibrium = solve_gripped_chain(chain, seed, _controls())
    first = gripped_chain_spectrum(chain, equilibrium.poses, _controls(), _scales())
    second = gripped_chain_spectrum(
        chain,
        equilibrium.poses,
        _controls(),
        replace(_scales(), length_m=0.1, time_s=0.05),
    )
    _assert_roots(first.rates_s_inv, second.rates_s_inv)
    np.testing.assert_allclose(
        first.grip_wrenches[0], equilibrium.grip_responses[0].root_wrench, atol=1e-12
    )
    assert first.frame.angular_velocity_rad_s == (10, 0, 0)
    assert first.stability_status == "unqualified"


def test_root_balance_and_strain_domain_are_checked_again() -> None:
    chain, poses = _model()
    poses[:, 2, 3] += 0.01
    with pytest.raises(ValueError, match="balance"):
        gripped_chain_spectrum(chain, poses, _controls(), _scales())
    poses[-1, 2, 3] += 0.2
    with pytest.raises(ValueError, match="strain"):
        gripped_chain_spectrum(chain, poses, _controls(), _scales())
