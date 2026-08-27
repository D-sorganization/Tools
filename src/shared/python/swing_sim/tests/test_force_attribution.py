"""Analytical contracts for coordinate-explicit pendulum force attribution."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.swing_sim.force_attribution import (
    DoublePendulumAttributionProvider,
    attribute_state,
    attribute_trajectory,
    component_impulse_objective,
)
from shared.python.swing_sim.types import PendulumParameters


def _provider() -> DoublePendulumAttributionProvider:
    return DoublePendulumAttributionProvider(
        PendulumParameters.golf_default(),
        g_inplane=(0.0, -9.80665),
    )


def test_double_pendulum_split_matches_closed_form_and_closes() -> None:
    provider = _provider()
    q = np.array([0.42, -0.73])
    velocity = np.array([3.2, -5.1])
    applied = np.array([18.0, -4.0])

    result = attribute_state(provider, q, velocity, applied)

    p = provider.parameters
    coupling = p.m2 * p.l1 * p.lc2 * np.sin(q[1])
    expected_coriolis_term = np.array(
        [-2.0 * coupling * velocity[0] * velocity[1], 0.0]
    )
    expected_squared_speed_term = np.array(
        [-coupling * velocity[1] ** 2, coupling * velocity[0] ** 2]
    )
    np.testing.assert_allclose(
        result.components["coriolis"].equation_term,
        expected_coriolis_term,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.components["squared_speed"].equation_term,
        expected_squared_speed_term,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.velocity_bias,
        expected_coriolis_term + expected_squared_speed_term,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sum(component.generalized_drive for component in result.components.values()),
        result.total_generalized_drive,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        provider.mass_matrix(q) @ result.acceleration,
        result.total_generalized_drive,
        atol=1e-12,
    )


def test_zero_velocity_removes_both_motion_dependent_components() -> None:
    result = attribute_state(
        _provider(), np.array([0.2, -0.4]), np.zeros(2), np.zeros(2)
    )

    for name in ("coriolis", "squared_speed", "damping", "velocity_residual"):
        np.testing.assert_array_equal(
            result.components[name].equation_term, np.zeros(2)
        )
    assert result.components["coriolis"].tangent_force_n is None


def test_force_only_hand_path_mapping_reports_unrepresented_joint_residual() -> None:
    result = attribute_state(
        _provider(),
        np.array([0.25, -0.55]),
        np.array([2.3, -4.0]),
        np.zeros(2),
    )

    component = result.components["squared_speed"]
    assert component.mapping_rank == 1
    assert component.mapping_status == "rank_deficient_force_only"
    assert component.endpoint_force_n.shape == (2,)
    assert abs(component.mapping_residual_nm[1]) > 0.0
    np.testing.assert_allclose(
        component.endpoint_generalized_drive_nm + component.mapping_residual_nm,
        component.generalized_drive,
        atol=1e-12,
    )


def test_trajectory_integrals_keep_impulse_power_and_work_distinct() -> None:
    time = np.linspace(0.0, 0.3, 7)
    q = np.column_stack((0.2 + 0.8 * time, -0.7 + 0.4 * time))
    velocity = np.tile(np.array([3.0, -4.0]), (time.size, 1))
    applied = np.tile(np.array([12.0, -2.0]), (time.size, 1))

    result = attribute_trajectory(_provider(), time, q, velocity, applied)
    metric = result.metrics["coriolis"]

    assert metric.signed_generalized_impulse_nm_s.shape == (2,)
    assert metric.absolute_generalized_impulse_nm_s.shape == (2,)
    assert metric.signed_tangent_impulse_n_s is not None
    assert metric.absolute_tangent_impulse_n_s is not None
    assert metric.generalized_work_j == pytest.approx(
        np.trapezoid(result.components["coriolis"].generalized_power_w, time)
    )
    assert metric.absolute_tangent_impulse_n_s >= abs(metric.signed_tangent_impulse_n_s)
    assert component_impulse_objective(result, "coriolis") == pytest.approx(
        -metric.signed_tangent_impulse_n_s
    )


def test_trajectory_integrates_only_defined_tangent_intervals() -> None:
    time = np.array([0.0, 0.1, 0.2, 0.3])
    q = np.tile(np.array([0.2, -0.7]), (time.size, 1))
    velocity = np.array([[0.0, 0.0], [2.0, -3.0], [2.0, -3.0], [0.0, 0.0]])

    result = attribute_trajectory(
        _provider(), time, q, velocity, np.zeros((time.size, 2))
    )
    metric = result.metrics["coriolis"]

    assert metric.signed_tangent_impulse_n_s is not None
    assert metric.tangent_valid_duration_s == pytest.approx(0.1)
    assert metric.tangent_total_duration_s == pytest.approx(0.3)


@pytest.mark.parametrize(
    ("time", "q", "velocity", "applied", "message"),
    [
        (np.array([0.0]), np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2)), "two"),
        (
            np.array([0.0, 0.0]),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            "strictly increasing",
        ),
        (
            np.array([0.0, 0.1]),
            np.zeros((2, 3)),
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            "shape",
        ),
    ],
)
def test_trajectory_contract_rejects_invalid_inputs(
    time: np.ndarray,
    q: np.ndarray,
    velocity: np.ndarray,
    applied: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        attribute_trajectory(_provider(), time, q, velocity, applied)
