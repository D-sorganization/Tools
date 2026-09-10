"""Work-map derivatives and bounded repeated algebra for kinetic quadrature."""

import numpy as np
import pytest

from shared.python.golf_club import _shaft_se3 as se3


@pytest.mark.parametrize("angle", [0.0, 1e-10, 0.4, 2.6])
@pytest.mark.parametrize("fraction", [0.0, 0.23, 1.0])
def test_joint_mapping_matches_directional_pose_map_derivative(
    angle: float, fraction: float
) -> None:
    relative = np.array([0.2, -0.1, 0.8, angle, 0, 0])
    direction = np.array([0.4, -0.3, 0.2, -0.2, 0.3, 0.5])
    mapping, rate = se3.section_velocity_kinematics(relative, fraction, direction)
    np.testing.assert_allclose(
        mapping, se3.section_velocity_map(relative, fraction), atol=2e-14
    )
    for step in (2e-6, 1e-6):
        oracle = (
            se3.section_velocity_map(relative + step * direction, fraction)
            - se3.section_velocity_map(relative - step * direction, fraction)
        ) / (2 * step)
        np.testing.assert_allclose(rate, oracle, rtol=1e-6, atol=8e-10)
    np.testing.assert_array_equal(rate[:, :6], -rate[:, 6:])


def test_joint_mapping_computes_each_frechet_pair_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    original = se3.expm_frechet

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("compute_expm"))
        return original(*args, **kwargs)

    def redundant(*args: object, **kwargs: object) -> None:
        raise AssertionError("joint kinetics must reuse the Frechet exponential")

    monkeypatch.setattr(se3, "expm_frechet", counted)
    monkeypatch.setattr(se3, "expm", redundant)
    se3.section_velocity_kinematics([0, 0, 1, 0.2, 0, 0], 0.3, np.ones(6))
    assert calls == [True, True]


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_endpoint_maps_are_exact_and_need_no_exponential(
    fraction: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unnecessary(*args: object, **kwargs: object) -> None:
        raise AssertionError("endpoint interpolation is the exact nodal map")

    monkeypatch.setattr(se3, "expm", unnecessary)
    monkeypatch.setattr(se3, "expm_frechet", unnecessary)
    mapping, rate = se3.section_velocity_kinematics(
        [0.1, -0.2, 0.5, 0.2, -0.4, 0.7], fraction, np.ones(6)
    )
    expected = np.zeros((6, 12))
    start = 6 * int(fraction)
    expected[:, start : start + 6] = np.eye(6)
    np.testing.assert_array_equal(mapping, expected)
    np.testing.assert_array_equal(rate, np.zeros((6, 12)))
    mapping[:] = 17
    assert np.all(rate == 0)


@pytest.mark.parametrize(
    "relative,fraction,direction",
    [
        ([0, 0, 1, np.pi, 0, 0], 0.0, np.ones(6)),
        (np.zeros(6), True, np.ones(6)),
        (np.zeros(6), 1.0, [0, 0, 0, 0, 0, np.nan]),
        ([False, 0, 0, 0, 0, 0], 0.0, np.ones(6)),
    ],
)
def test_endpoint_shortcut_retains_all_input_refusals(
    relative: object, fraction: object, direction: object
) -> None:
    with pytest.raises((ValueError, TypeError)):
        se3.section_velocity_kinematics(relative, fraction, direction)
