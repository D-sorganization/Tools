"""Independent damped-polynomial and instability controls, Tools #5072."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club._shaft_damped_spectrum import (
    DampedPencil,
    frozen_damped_spectrum,
)
from shared.python.golf_club._shaft_spectrum import _frozen_spectrum

from .test_shaft_spectrum import _assert_roots, _scales


@pytest.mark.parametrize("damping", [0.0, 0.1, 3.0])
def test_scalar_damping_preserves_both_physical_characteristic_roots(
    damping: float,
) -> None:
    pencil = DampedPencil([[1]], [[0]], [[damping]], [[1]])
    result = frozen_damped_spectrum(pencil, _scales())
    _assert_roots(result.rates_s_inv, np.roots([1, damping, 1]))
    np.testing.assert_allclose(
        result.velocity_modes,
        result.displacement_modes * result.rates_s_inv,
        atol=1e-12,
    )
    assert max(result.polynomial_relative_residuals) < 1e-13
    assert result.stability_status == "unqualified"


def test_critical_damping_reports_its_ill_conditioned_modal_basis() -> None:
    result = frozen_damped_spectrum(DampedPencil([[1]], [[0]], [[2]], [[1]]), _scales())
    np.testing.assert_allclose(result.rates_s_inv, -1, atol=2e-7)
    assert result.scaled_eigenbasis_rcond < 1e-7
    assert result.stability_status == "unqualified"


def test_positive_damping_does_not_remove_negative_stiffness_growth() -> None:
    result = frozen_damped_spectrum(
        DampedPencil([[1]], [[0]], [[1]], [[-1]]), _scales()
    )
    _assert_roots(
        result.rates_s_inv, np.array([(-1 + np.sqrt(5)) / 2, (-1 - np.sqrt(5)) / 2])
    )


def test_passive_damping_can_destabilize_a_gyroscopic_equilibrium() -> None:
    mass, gyro, stiffness = np.eye(2), np.array([[0, -3.0], [3.0, 0]]), -np.eye(2)
    undamped = _frozen_spectrum(mass, gyro, stiffness, _scales())
    zero = frozen_damped_spectrum(
        DampedPencil(mass, gyro, np.zeros((2, 2)), stiffness), _scales()
    )
    _assert_roots(zero.rates_s_inv, undamped.rates_s_inv)
    damped = frozen_damped_spectrum(
        DampedPencil(mass, gyro, 0.1 * np.eye(2), stiffness), _scales()
    )
    # det(s^2 I + (0.1 I+3J)s - I)=(s^2+0.1s-1)^2+9s^2.
    expected = np.roots([1, 0.2, 7.01, -0.2, 1])
    _assert_roots(damped.rates_s_inv, expected)
    assert np.count_nonzero(damped.rates_s_inv.real > 0.017) == 2
    assert damped.stability_status == "unqualified"


def test_semidefinite_loss_retains_an_undamped_mode() -> None:
    pencil = DampedPencil(
        np.eye(2), np.zeros((2, 2)), np.diag([0.2, 0]), np.diag([1, 4])
    )
    result = frozen_damped_spectrum(pencil, _scales())
    _assert_roots(result.rates_s_inv, np.r_[np.roots([1, 0.2, 1]), [2j, -2j]])


def test_nonsymmetric_stiffness_is_retained_without_projection() -> None:
    pencil = DampedPencil(
        np.eye(2), np.zeros((2, 2)), 0.1 * np.eye(2), [[4, 1], [-1, 4]]
    )
    result = frozen_damped_spectrum(pencil, _scales())
    expected = np.r_[np.roots([1, 0.1, 4 + 1j]), np.roots([1, 0.1, 4 - 1j])]
    _assert_roots(result.rates_s_inv, expected)


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("mass", [[1, 1], [0, 1]], "mass"),
        ("mass", [[1, 0], [0, 0]], "mass"),
        ("mass", [[1, 0], [0, -1]], "mass"),
        ("mass", [[1, 0], [0, 1e-15]], "mass"),
        ("gyroscopic", [[1, 0], [0, 0]], "gyroscopic"),
        ("damping", [[1, 1], [0, 1]], "damping"),
        ("damping", [[1, 2], [2, 1]], "damping"),
    ],
)
def test_invalid_mechanical_coefficients_are_refused(
    name: str,
    value: object,
    message: str,
) -> None:
    pencil = DampedPencil(np.eye(2), np.zeros((2, 2)), np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match=message):
        frozen_damped_spectrum(replace(pencil, **{name: value}), _scales())


@pytest.mark.parametrize("value", [[], [[1, 2]], [[np.inf]], [[True]], [[1j]]])
def test_pencil_rejects_malformed_or_nonreal_arrays(value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        DampedPencil(value, [[0]], [[0]], [[1]])


def test_inputs_are_owned_and_time_scaling_preserves_physical_roots() -> None:
    mass = np.eye(2)
    pencil = DampedPencil(mass, np.zeros((2, 2)), np.diag([0.1, 0.3]), np.diag([4, 9]))
    mass[:] = 0
    first = frozen_damped_spectrum(pencil, _scales())
    second = frozen_damped_spectrum(pencil, replace(_scales(), time_s=0.4))
    _assert_roots(first.rates_s_inv, second.rates_s_inv)
    assert first.scaled_eigenbasis_rcond != pytest.approx(
        second.scaled_eigenbasis_rcond
    )


def test_numerical_failure_and_bad_eigenpairs_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pencil = DampedPencil([[1]], [[0]], [[0.1]], [[1]])
    with pytest.raises(ValueError, match="scale"):
        frozen_damped_spectrum(pencil, replace(_scales(), time_s=1e-200))
    with pytest.raises(ValueError, match="numerical"):
        frozen_damped_spectrum(pencil, replace(_scales(), time_s=1e200))
    with pytest.raises(TypeError, match="SpectrumScales"):
        frozen_damped_spectrum(pencil, None)
    monkeypatch.setattr(np.linalg, "eig", lambda _: (np.array([1.0, -1.0]), np.eye(2)))
    with pytest.raises(ValueError, match="residual"):
        frozen_damped_spectrum(pencil, _scales())
