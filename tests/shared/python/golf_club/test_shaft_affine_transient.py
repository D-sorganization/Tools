"""Analytic and independent integration oracles for constant affine motion."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from shared.python.golf_club import _shaft_affine_transient as transient
from shared.python.golf_club._shaft_damped_spectrum import DampedPencil
from shared.python.golf_club._shaft_spectrum import SpectrumScales


def _scales(time: float = 1.0) -> SpectrumScales:
    return SpectrumScales(1.0, time, 1e-12, 1e-9)


def _pencil(stiffness: float = 1, damping: float = 0) -> DampedPencil:
    return DampedPencil([[1]], [[0]], [[damping]], [[stiffness]])


@pytest.mark.parametrize("time_scale", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("time", [0.0, 0.1, 1.0, 3.0])
def test_constant_residual_critical_damping_has_analytic_step_response(
    time_scale: float, time: float
) -> None:
    state = transient.affine_state_at(
        _pencil(damping=2), [-3], _scales(time_scale), [0, 0], time
    )
    expected = [
        3 * (1 - (1 + time) * np.exp(-time)),
        time_scale * 3 * time * np.exp(-time),
    ]
    np.testing.assert_allclose(state, expected, atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize("time", [0.0, 0.3, 2.0])
def test_neutral_defective_free_particle_requires_no_stiffness_inverse(
    time: float,
) -> None:
    state = transient.affine_state_at(_pencil(0), [-4], _scales(), [2, 3], time)
    np.testing.assert_allclose(state, [2 + 3 * time + 2 * time**2, 3 + 4 * time])


def test_growing_motion_is_retained_and_not_reported_as_stable() -> None:
    state = transient.affine_state_at(_pencil(-4), [0], _scales(), [1, 0], 1.3)
    np.testing.assert_allclose(state, [np.cosh(2.6), 2 * np.sinh(2.6)], rtol=1e-13)


def test_undamped_oscillator_preserves_analytic_motion_and_energy() -> None:
    times = np.linspace(0, 5, 31)
    states = np.array(
        [
            transient.affine_state_at(_pencil(9), [0], _scales(), [2, -3], t)
            for t in times
        ]
    )
    np.testing.assert_allclose(states[:, 0], 2 * np.cos(3 * times) - np.sin(3 * times))
    np.testing.assert_allclose(
        states[:, 1], -6 * np.sin(3 * times) - 3 * np.cos(3 * times)
    )
    np.testing.assert_allclose((9 * states[:, 0] ** 2 + states[:, 1] ** 2) / 2, 22.5)


def test_coupled_gyroscopic_circulatory_system_matches_independent_ivp() -> None:
    mass = np.array([[2, 0.2], [0.2, 1]])
    gyro = np.array([[0, 0.7], [-0.7, 0]])
    damping = np.diag([0.3, 0.4])
    stiffness = np.array([[7, -2], [-1, 3]])
    load = np.array([0.3, -0.2])
    initial = np.array([0.2, -0.1, 0.4, 0.7])

    def physical_rhs(_time: float, state: np.ndarray) -> np.ndarray:
        acceleration = np.linalg.solve(
            mass, -load - (gyro + damping) @ state[2:] - stiffness @ state[:2]
        )
        return np.r_[state[2:], acceleration]

    reference = solve_ivp(
        physical_rhs, (0, 1.7), initial, method="DOP853", rtol=1e-12, atol=1e-14
    )
    assert reference.success
    pencil = DampedPencil(mass, gyro, damping, stiffness)
    for scale in (0.2, 1.0, 3.0):
        state = transient.affine_state_at(
            pencil, load, _scales(scale), initial * [1, 1, scale, scale], 1.7
        )
        np.testing.assert_allclose(
            np.array(state) / [1, 1, scale, scale], reference.y[:, -1], atol=2e-12
        )


@pytest.mark.parametrize("time", [-1, True, "1", 1j, np.nan, np.inf, [1]])
def test_time_domain_is_strict(time: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        transient.affine_state_at(_pencil(), [0], _scales(), [1, 0], time)


@pytest.mark.parametrize("state", [[1], [1, np.nan], [True, 0], [1j, 0], ["1", "0"]])
def test_state_domain_is_strict(state: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        transient.affine_state_at(_pencil(), [0], _scales(), state, 0)


def test_zero_time_returns_owned_state_and_still_validates_plant_and_load() -> None:
    original = np.array([1.2, -0.4])
    result = transient.affine_state_at(_pencil(), [0], _scales(), original, 0)
    original[:] = 0
    assert result == (1.2, -0.4)
    for pencil, load in (
        (_pencil(), [np.nan]),
        (DampedPencil([[0]], [[0]], [[0]], [[1]]), [0]),
    ):
        with pytest.raises(ValueError):
            transient.affine_state_at(pencil, load, _scales(), [0, 0], 0)


def test_affine_semigroup_retains_residual_in_both_intervals() -> None:
    pencil = _pencil(4, 0.3)
    first = transient.affine_state_at(pencil, [-2], _scales(), [1, 0], 0.3)
    split = transient.affine_state_at(pencil, [-2], _scales(), first, 0.7)
    direct = transient.affine_state_at(pencil, [-2], _scales(), [1, 0], 1)
    np.testing.assert_allclose(split, direct, rtol=1e-13, atol=1e-14)


def test_overflow_is_refused_instead_of_returning_nonfinite_motion() -> None:
    with pytest.raises(ValueError):
        transient.affine_state_at(_pencil(-1), [0], _scales(), [1, 0], 1000)


def test_underflow_cannot_silently_turn_a_positive_time_into_zero_time() -> None:
    with pytest.raises(ValueError, match="time|evaluation"):
        transient.affine_state_at(_pencil(0), [0], _scales(1e100), [0, 1e100], 1e-250)


def test_representable_tiny_residual_motion_is_not_zeroed() -> None:
    state = transient.affine_state_at(_pencil(0), [-1e-200], _scales(), [0, 0], 0.5)
    np.testing.assert_allclose(state, [1.25e-201, 5e-201], atol=0, rtol=1e-14)


def test_gripped_adapter_uses_its_owned_pencil_residual_and_scaling() -> None:
    from shared.python.golf_club._shaft_gripped_operating import constant_gripped_model

    from .test_shaft_equilibrium import _controls
    from .test_shaft_gripped_operating import _loaded_model

    chain, poses = _loaded_model()
    model = constant_gripped_model(chain, poses, _controls(), _scales(0.2))
    initial = np.zeros(24)
    initial[8] = 1e-4
    expected = transient.affine_state_at(
        model.pencil, model.scaled_residual, model.scales, initial, 0.05
    )
    assert model.scaled_state_at(initial, 0.05) == expected
    assert model.stability_status == "unqualified"
