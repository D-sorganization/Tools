"""Independent matrix-exponential oracles for bounded SE(3) evaluation."""

import numpy as np
import pytest
from scipy.linalg import expm

from shared.python.golf_club import _shaft_se3 as se3


def _block_oracle(
    relative: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    generator = np.zeros((12, 12))
    generator[:6, :6] = -se3.twist_ad(relative)
    generator[:6, 6:] = np.eye(6)
    change = np.zeros((12, 12))
    change[:6, :6] = -se3.twist_ad(direction)
    block = np.block([[generator, change], [np.zeros((12, 12)), generator]])
    evaluated = expm(block)
    return evaluated[:6, 6:12], evaluated[:6, 18:24]


@pytest.mark.parametrize("angle", [0.0, 1e-10, 0.4, 2.6, np.pi - 2e-6])
@pytest.mark.parametrize("scale", [1e-120, 1.0, 1e120])
def test_polynomial_pair_matches_independent_augmented_exponential(
    angle: float, scale: float
) -> None:
    relative = np.array([0.1, -0.2, 0.3, angle, 0, 0])
    direction = np.array([0.4, -0.3, 0.2, -0.2, 0.3, 0.5])
    expected, derivative = _block_oracle(relative, direction)
    actual = se3._polynomial_jacobian_pair(relative, scale * direction)
    assert actual is not None
    np.testing.assert_allclose(actual[0], expected, rtol=3e-14, atol=2e-14)
    np.testing.assert_allclose(actual[1] / scale, derivative, rtol=3e-14, atol=2e-14)


def test_zero_twist_has_exact_owned_identity_and_half_direction() -> None:
    relative = np.zeros(6)
    direction = np.arange(6, dtype=float)
    result = se3._polynomial_jacobian_pair(relative, direction)
    assert result is not None
    np.testing.assert_array_equal(result[0], np.eye(6))
    np.testing.assert_array_equal(result[1], -se3.twist_ad(direction) / 2)
    result[0][:] = 17
    result[1][:] = 17
    np.testing.assert_array_equal(relative, np.zeros(6))
    np.testing.assert_array_equal(direction, np.arange(6))


@pytest.mark.parametrize("seed", range(12))
def test_arbitrary_axes_and_noncommuting_directions(seed: int) -> None:
    generator = np.random.default_rng(seed)
    relative = generator.normal(size=6)
    relative *= 3.9 / np.linalg.norm(se3.twist_ad(relative), ord=np.inf)
    relative *= min(1.0, 3.0 / float(np.linalg.norm(relative[3:])))
    direction = generator.normal(size=6)
    actual = se3._polynomial_jacobian_pair(relative, direction)
    assert actual is not None
    expected = _block_oracle(relative, direction)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=3e-14, atol=2e-14)


def test_bounded_joint_pair_needs_no_general_exponential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unnecessary(*args: object, **kwargs: object) -> None:
        raise AssertionError("bounded polynomial needs no general exponential")

    monkeypatch.setattr(se3, "expm", unnecessary)
    monkeypatch.setattr(se3, "expm_frechet", unnecessary)
    actual = se3._jacobian_pair(np.array([0, 0, 1, 0.2, 0, 0]), np.ones(6))
    assert all(np.all(np.isfinite(value)) for value in actual)


@pytest.mark.parametrize("relative", [[5, 0, 0, 0.2, 0, 0], [0, 0, 0, 3.2, 0, 0]])
def test_outside_polynomial_domain_retains_the_general_routine(
    relative: list[float], monkeypatch: pytest.MonkeyPatch
) -> None:
    twist, direction = np.array(relative), np.ones(6)
    original = se3.expm_frechet
    calls = []

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("compute_expm"))
        return original(*args, **kwargs)

    monkeypatch.setattr(se3, "expm_frechet", counted)
    assert se3._polynomial_jacobian_pair(twist, direction) is None
    actual = se3._jacobian_pair(twist, direction)
    assert calls == [True]
    for result, reference in zip(actual, _block_oracle(twist, direction), strict=True):
        np.testing.assert_allclose(result, reference, rtol=3e-14, atol=2e-14)


@pytest.mark.parametrize("zero_direction", [False, True])
def test_public_derivative_uses_bounded_or_exact_zero_evaluation(
    zero_direction: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = np.array([5 if zero_direction else 1, 0, 0, 0.2, 0, 0])
    direction = np.zeros(6) if zero_direction else np.ones(6)
    expected = _block_oracle(relative, direction)[1]

    def unnecessary(*args: object, **kwargs: object) -> None:
        raise AssertionError("this derivative needs no general exponential")

    monkeypatch.setattr(se3, "expm_frechet", unnecessary)
    actual = se3.right_jacobian_derivative(relative, direction)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=2e-14)


@pytest.mark.parametrize(
    "bad", [np.zeros(5), np.full(6, np.nan), np.ones(6, dtype=bool)]
)
def test_zero_direction_does_not_bypass_twist_contracts(bad: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        se3.right_jacobian_derivative(bad, np.zeros(6))
