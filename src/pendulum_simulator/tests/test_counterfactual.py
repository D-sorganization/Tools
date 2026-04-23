"""
Tests for zero-torque counterfactual joint force computation.

TDD: these tests define the expected behaviour BEFORE implementation.

Design by Contract
------------------
- zero_torque_joint_forces_double requires state.shape == (4,), finite values
- zero_torque_joint_forces_triple requires state.shape == (6,), finite values
- All outputs are finite dicts with the expected joint keys
- Counterfactual forces ≠ actual forces when nonzero torque is applied
"""

from __future__ import annotations

import numpy as np
import pytest

from double_pendulum_golf.physics import PendulumParams
from double_pendulum_golf.physics_triple import TriplePendulumParams
from double_pendulum_golf.counterfactual import (
    zero_torque_joint_forces_double,
    zero_torque_joint_forces_triple,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def double_params() -> PendulumParams:
    return PendulumParams(m1=5.0, m2=0.5, L1=0.6, L2=1.0, g=9.81)


@pytest.fixture()
def triple_params() -> TriplePendulumParams:
    return TriplePendulumParams(m1=5.0, m2=0.5, m3=0.3, L1=0.6, L2=1.0, L3=0.5, g=9.81)


@pytest.fixture()
def double_state() -> np.ndarray:
    """Double pendulum state: [theta1, phi, dtheta1, dphi] in radians."""
    return np.array([np.radians(45.0), np.radians(-30.0), 0.5, -0.3])


@pytest.fixture()
def triple_state() -> np.ndarray:
    """Triple pendulum state: [theta1, phi1, phi2, dtheta1, dphi1, dphi2]."""
    return np.array(
        [np.radians(45.0), np.radians(-30.0), np.radians(20.0), 0.5, -0.3, 0.2]
    )


# ---------------------------------------------------------------------------
# Double pendulum counterfactual
# ---------------------------------------------------------------------------


class TestZeroTorqueDouble:
    def test_returns_shoulder_and_wrist_keys(
        self, double_state: np.ndarray, double_params: PendulumParams
    ) -> None:
        """Output dict must have 'shoulder' and 'wrist' keys."""
        result = zero_torque_joint_forces_double(double_state, double_params)
        assert "shoulder" in result
        assert "wrist" in result

    def test_forces_are_2tuples_of_finite_floats(
        self, double_state: np.ndarray, double_params: PendulumParams
    ) -> None:
        """Each joint force must be a finite (fx, fy) pair."""
        result = zero_torque_joint_forces_double(double_state, double_params)
        for key in ("shoulder", "wrist"):
            fx, fy = result[key]
            assert np.isfinite(fx), f"{key} fx is not finite"
            assert np.isfinite(fy), f"{key} fy is not finite"

    def test_zero_velocity_zero_torque_matches_static_gravity(
        self, double_params: PendulumParams
    ) -> None:
        """At rest with gravity, shoulder force ~ (m1+m2)*g upward."""
        state = np.array([0.0, 0.0, 0.0, 0.0])  # hanging straight down
        result = zero_torque_joint_forces_double(state, double_params)
        fx, fy = result["shoulder"]
        expected_fy = (double_params.m1 + double_params.m2) * double_params.g
        assert abs(fx) < 1e-8, f"No horizontal force at rest, got fx={fx}"
        assert (
            abs(fy - expected_fy) < 1e-4
        ), f"Shoulder fy={fy:.4f}, expected {expected_fy:.4f}"

    def test_differs_from_driven_forces_when_torque_nonzero(
        self, double_state: np.ndarray, double_params: PendulumParams
    ) -> None:
        """Counterfactual forces must differ from driven forces with torque != 0."""
        from double_pendulum_golf.simulation import SimulationResult

        # Create a fake result with large driving torque

        def torque_func(t: float) -> tuple[float, float]:
            return (50.0, 20.0)

        t_arr = np.array([0.0])
        result = SimulationResult(
            t=t_arr,
            states=double_state[None, :],
            params=double_params,
            torque_func=torque_func,
        )
        actual = result.joint_forces_at(0)
        counterfactual = zero_torque_joint_forces_double(double_state, double_params)

        # With 50 Nm at shoulder, forces should differ meaningfully
        diff_shoulder = abs(actual["shoulder"][1] - counterfactual["shoulder"][1])
        assert (
            diff_shoulder > 1.0
        ), f"Expected driven vs zero-torque to differ; got diff={diff_shoulder:.3f}"

    def test_zero_gravity_hanging_position(self, double_params: PendulumParams) -> None:
        """With g=0, zero-torque counterfactual gives near-zero forces at rest."""
        params_no_g = PendulumParams(
            m1=double_params.m1,
            m2=double_params.m2,
            L1=double_params.L1,
            L2=double_params.L2,
            g=0.0,
        )
        state = np.array([0.0, 0.0, 0.0, 0.0])
        result = zero_torque_joint_forces_double(state, params_no_g)
        for key in ("shoulder", "wrist"):
            fx, fy = result[key]
            assert (
                abs(fx) < 1e-8 and abs(fy) < 1e-8
            ), f"No gravity + no motion → zero force at {key}, got ({fx:.2e},{fy:.2e})"

    def test_invalid_state_shape_raises(self, double_params: PendulumParams) -> None:
        """Non-(4,) state must raise AssertionError."""
        with pytest.raises((ValueError, TypeError)):
            zero_torque_joint_forces_double(np.zeros(3), double_params)

    def test_nonfinite_state_raises(self, double_params: PendulumParams) -> None:
        """NaN state must raise AssertionError."""
        with pytest.raises((ValueError, TypeError)):
            zero_torque_joint_forces_double(
                np.array([np.nan, 0.0, 0.0, 0.0]), double_params
            )


# ---------------------------------------------------------------------------
# Triple pendulum counterfactual
# ---------------------------------------------------------------------------


class TestZeroTorqueTriple:
    def test_returns_all_joint_keys(
        self, triple_state: np.ndarray, triple_params: TriplePendulumParams
    ) -> None:
        """Output must have 'shoulder', 'wrist1', 'wrist2' keys."""
        result = zero_torque_joint_forces_triple(triple_state, triple_params)
        assert "shoulder" in result
        assert "wrist1" in result
        assert "wrist2" in result

    def test_forces_are_finite(
        self, triple_state: np.ndarray, triple_params: TriplePendulumParams
    ) -> None:
        result = zero_torque_joint_forces_triple(triple_state, triple_params)
        for key in ("shoulder", "wrist1", "wrist2"):
            fx, fy = result[key]
            assert np.isfinite(fx) and np.isfinite(
                fy
            ), f"{key} forces not finite: ({fx}, {fy})"

    def test_static_hanging_shoulder_force(
        self, triple_params: TriplePendulumParams
    ) -> None:
        """At rest hanging straight down, shoulder force ≈ (m1+m2+m3)*g."""
        state = np.zeros(6)
        result = zero_torque_joint_forces_triple(state, triple_params)
        fx, fy = result["shoulder"]
        expected_fy = (
            triple_params.m1 + triple_params.m2 + triple_params.m3
        ) * triple_params.g
        assert abs(fx) < 1e-8
        assert abs(fy - expected_fy) < 1e-4, f"fy={fy}, expected {expected_fy}"

    def test_invalid_state_shape_raises(
        self, triple_params: TriplePendulumParams
    ) -> None:
        with pytest.raises((ValueError, TypeError)):
            zero_torque_joint_forces_triple(np.zeros(5), triple_params)
