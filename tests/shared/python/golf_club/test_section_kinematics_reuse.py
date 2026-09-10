"""Bound repeated exponential work without changing section-inertia physics."""

import numpy as np
import pytest

from shared.python.golf_club import _shaft_se3 as se3
from shared.python.golf_club._shaft_inertia import SectionInertia

from .test_shaft_inertia import _poses, _samples


@pytest.mark.parametrize("order", [2, 4])
def test_inertia_computes_one_full_frechet_pair_per_evaluation(
    order: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    original = se3.expm_frechet

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("compute_expm"))
        return original(*args, **kwargs)

    monkeypatch.setattr(se3, "expm_frechet", counted)
    model = SectionInertia(_samples(order))
    velocity = np.linspace(-0.3, 0.4, 12)
    first = model.evaluate(_poses(), velocity)
    assert calls == [True] * (order + 1)  # one full pair plus each fraction
    second = model.evaluate(_poses(), velocity)
    assert calls == [True] * (2 * (order + 1))  # no cache across state evaluations
    np.testing.assert_array_equal(first.mass, second.mass)
    np.testing.assert_array_equal(first.bias, second.bias)
    np.testing.assert_array_equal(first.mass_rate, second.mass_rate)


def test_prepared_kinematics_owns_inputs_and_returns_independent_maps() -> None:
    relative = np.array([0.2, -0.1, 0.8, 0.4, -0.1, 0.2])
    direction = np.array([0.4, -0.3, 0.2, -0.2, 0.3, 0.5])
    expected = se3.section_velocity_kinematics(relative, 0.23, direction)
    prepared = se3._SectionVelocityKinematics(relative, direction)
    relative[:] = 0
    direction[:] = 17
    actual = prepared.at(0.23)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, reference)
        result[:] = 99
    again = prepared.at(0.23)
    for result, reference in zip(again, expected, strict=True):
        np.testing.assert_array_equal(result, reference)


@pytest.mark.parametrize("angle", [0.0, 1e-10, 0.4, 2.6])
def test_relative_inverse_maps_match_both_direct_exponentials(angle: float) -> None:
    relative = np.array([0.2, -0.1, 0.8, angle, 0, 0])
    expected_left = np.linalg.solve(se3.right_jacobian(-relative), np.eye(6))
    expected_right = np.linalg.solve(se3.right_jacobian(relative), np.eye(6))
    mapping, left, right = se3._relative_maps(relative)
    np.testing.assert_allclose(left, expected_left, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(right, expected_right, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(
        mapping, np.hstack((-expected_left, expected_right)), rtol=2e-13, atol=2e-14
    )


def test_relative_inverse_maps_evaluate_one_jacobian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = se3.right_jacobian
    calls = []

    def counted(value: object) -> np.ndarray:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(se3, "right_jacobian", counted)
    se3._relative_maps(np.array([0.2, -0.1, 0.8, 0.4, -0.2, 0.1]))
    assert len(calls) == 1
