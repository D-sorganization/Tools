"""Tests for GPU-accelerated torque profile optimization.

Verifies gradient correctness and convergence behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if JAX not available
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

from double_pendulum_golf.physics_golfer import N_DOF  # noqa: E402
from double_pendulum_golf.physics_golfer_jax import GolferParamsJAX  # noqa: E402
from double_pendulum_golf.optimizer_gpu import (  # noqa: E402
    clubhead_speed_objective,
    clubhead_velocity_at_final_time,
    compute_gradient_via_finite_difference,
    optimize_simple_torque_profile,
)
from double_pendulum_golf.constraint_solver import project_to_constraints  # noqa: E402

# Test parameters
_PARAMS = GolferParamsJAX(
    m_hub=2.0,
    m_r_upper=3.0,
    m_r_fore=2.0,
    m_l_upper=3.0,
    m_l_fore=2.0,
    m_club=0.5,
    L_hub=0.15,
    L_r_upper=0.35,
    L_r_fore=0.30,
    L_l_upper=0.35,
    L_l_fore=0.30,
    L_club=1.1,
    d_rs=0.20,
    d_ls=0.20,
    grip_right=0.05,
    grip_left=0.25,
    m_clubhead=0.2,
)


@pytest.fixture
def initial_state() -> np.ndarray:
    """Hanging initial state with zero velocity."""
    state = np.zeros(2 * N_DOF)
    # Project to constraint surface
    q = project_to_constraints(state[:N_DOF], _PARAMS)
    state[:N_DOF] = q
    return state


@pytest.fixture
def torque_coeffs() -> np.ndarray:
    """Sample torque coefficients."""
    return np.array([0.1, -0.05, 0.15, -0.1, 0.08, -0.12, 0.06])


class TestObjectiveFunction:
    """Test the objective function computation."""

    @pytest.mark.slow
    def test_objective_is_negative_speed(
        self, initial_state: np.ndarray, torque_coeffs: np.ndarray
    ) -> None:
        """Objective returns negative speed."""
        torque_jax = jnp.array(torque_coeffs)
        state_jax = jnp.array(initial_state)

        objective = clubhead_speed_objective(
            torque_jax, _PARAMS, state_jax, t_end=0.5, dt=0.01
        )

        # Objective should be a scalar
        assert objective.shape == ()

        # For positive torques starting from rest, speed should be negative
        # (since we minimize negative speed)
        assert float(objective) < 0


class TestGradientCorrectness:
    """Test gradient computation via autodiff vs finite differences."""

    @pytest.mark.slow
    def test_gradient_via_autodiff_vs_finite_difference(
        self, initial_state: np.ndarray, torque_coeffs: np.ndarray
    ) -> None:
        """JAX autodiff gradient matches finite differences."""
        torque_jax = jnp.array(torque_coeffs)
        state_jax = jnp.array(initial_state)

        # Compute gradient via autodiff
        def loss_fn(coeffs):
            return clubhead_speed_objective(coeffs, _PARAMS, state_jax, t_end=0.5, dt=0.01)

        grad_autodiff = jax.grad(loss_fn)(torque_jax)

        # Compute gradient via finite differences
        grad_fd = compute_gradient_via_finite_difference(
            torque_jax, _PARAMS, state_jax, t_end=0.5, eps=1e-4, dt=0.01
        )

        # Check that gradients are reasonably close
        # Note: tolerance is loose because of numerical integration errors
        grad_autodiff_np = np.array(grad_autodiff)
        grad_fd_np = np.array(grad_fd)

        # Normalize by max absolute value to avoid scale issues
        max_grad = np.max(np.abs(grad_fd_np))
        if max_grad > 1e-10:
            rel_error = np.linalg.norm(grad_autodiff_np - grad_fd_np) / (max_grad + 1e-12)
            assert rel_error < 0.5, f"Relative error in gradient: {rel_error}"


class TestOptimization:
    """Test convergence of optimization."""

    @pytest.mark.slow
    def test_optimization_reduces_loss(self, initial_state: np.ndarray) -> None:
        """Optimization reduces the loss function."""
        state_jax = jnp.array(initial_state)

        initial_torques = jnp.zeros(7)

        # Compute initial objective
        obj_initial = clubhead_speed_objective(
            initial_torques, _PARAMS, state_jax, t_end=0.5, dt=0.01
        )

        # Run short optimization
        optimal_torques, history = optimize_simple_torque_profile(
            _PARAMS,
            state_jax,
            t_end=0.5,
            n_iterations=5,
            learning_rate=0.01,
            dt=0.01,
            seed=42,
        )

        obj_final = clubhead_speed_objective(
            optimal_torques, _PARAMS, state_jax, t_end=0.5, dt=0.01
        )

        # Loss should decrease (objective should become more negative)
        assert float(obj_final) < float(obj_initial)

    @pytest.mark.slow
    def test_optimization_history_is_monotonic(self, initial_state: np.ndarray) -> None:
        """Optimization loss history is monotonic (mostly)."""
        state_jax = jnp.array(initial_state)

        _, history = optimize_simple_torque_profile(
            _PARAMS,
            state_jax,
            t_end=0.5,
            n_iterations=10,
            learning_rate=0.01,
            dt=0.01,
            seed=42,
        )

        # History should have 10 entries
        assert len(history) == 10

        # General trend should be decreasing (allowing some noise)
        # Use linear regression to check trend
        x = np.arange(len(history))
        y = np.array(history)
        slope = np.polyfit(x, y, 1)[0]
        # Slope should be negative (loss decreasing)
        assert slope < 0, f"Optimization not improving. Slope: {slope}"


class TestClubheadSpeed:
    """Test clubhead speed computation."""

    @pytest.mark.slow
    def test_clubhead_speed_is_positive(
        self, initial_state: np.ndarray, torque_coeffs: np.ndarray
    ) -> None:
        """Clubhead speed is always positive."""
        torque_jax = jnp.array(torque_coeffs)
        state_jax = jnp.array(initial_state)

        speed = clubhead_velocity_at_final_time(
            torque_jax, _PARAMS, state_jax, t_end=0.5, dt=0.01
        )

        assert float(speed) >= 0.0

    @pytest.mark.slow
    def test_clubhead_speed_increases_with_torque(self, initial_state: np.ndarray) -> None:
        """Clubhead speed is higher with positive torques."""
        state_jax = jnp.array(initial_state)

        # Zero torques
        speed_zero = clubhead_velocity_at_final_time(
            jnp.zeros(7), _PARAMS, state_jax, t_end=0.5, dt=0.01
        )

        # Positive torques
        torques_pos = jnp.array([0.2, 0.2, 0.2, 0.0, 0.2, 0.2, 0.0])
        speed_pos = clubhead_velocity_at_final_time(
            torques_pos, _PARAMS, state_jax, t_end=0.5, dt=0.01
        )

        # Positive torques should produce higher speed
        assert float(speed_pos) > float(speed_zero)


class TestFiniteElementGradient:
    """Test finite difference gradient computation."""

    @pytest.mark.slow
    def test_fd_gradient_shape(
        self, initial_state: np.ndarray, torque_coeffs: np.ndarray
    ) -> None:
        """Finite difference gradient has correct shape."""
        torque_jax = jnp.array(torque_coeffs)
        state_jax = jnp.array(initial_state)

        grad = compute_gradient_via_finite_difference(
            torque_jax, _PARAMS, state_jax, t_end=0.5, eps=1e-4, dt=0.01
        )

        assert grad.shape == (7,)

    @pytest.mark.slow
    def test_fd_gradient_is_finite(
        self, initial_state: np.ndarray, torque_coeffs: np.ndarray
    ) -> None:
        """Finite difference gradient produces finite values."""
        torque_jax = jnp.array(torque_coeffs)
        state_jax = jnp.array(initial_state)

        grad = compute_gradient_via_finite_difference(
            torque_jax, _PARAMS, state_jax, t_end=0.5, eps=1e-4, dt=0.01
        )

        grad_np = np.array(grad)
        assert np.all(np.isfinite(grad_np)), f"Gradient has non-finite values: {grad_np}"
