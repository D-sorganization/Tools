"""Tests for the constraint solver module.

Validates Baumgarte stabilization, KKT system, constraint projection,
and velocity projection.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import double_pendulum_golf.constraint_solver as constraint_solver_module
from double_pendulum_golf.constraint_solver import (
    constrained_accelerations,
    constraint_forces,
    constraint_violation,
    equations_of_motion,
    project_to_constraints,
    project_velocity,
)
from double_pendulum_golf.physics_golfer import (
    N_CONSTRAINTS,
    N_DOF,
    GolferParams,
    constraint_jacobian,
    constraint_vector,
)


@pytest.fixture
def golfer_params() -> GolferParams:
    """Symmetric golfer parameters for testing."""
    return GolferParams(
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
def zero_torque() -> Callable[[float], tuple[float, float, float, float, float, float, float]]:
    """Zero torque function for all joints."""
    return lambda t: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _make_consistent_state(params: GolferParams) -> np.ndarray:
    """Create a state that approximately satisfies constraints."""
    q = np.zeros(N_DOF)
    q = project_to_constraints(q, params)
    qdot = np.zeros(N_DOF)
    qdot = project_velocity(q, qdot, params)
    return np.concatenate([q, qdot])


class TestProjectToConstraints:
    """Constraint projection must drive violation to zero."""

    def test_zero_config_projects(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        q_proj = project_to_constraints(q, golfer_params)
        phi = constraint_vector(q_proj, golfer_params)
        assert np.linalg.norm(phi) < 1e-6, (
            f"Constraint violation after projection: {np.linalg.norm(phi)}"
        )

    def test_arbitrary_config_projects(self, golfer_params: GolferParams) -> None:
        rng = np.random.default_rng(123)
        q = rng.uniform(-0.5, 0.5, size=N_DOF)
        q_proj = project_to_constraints(q, golfer_params)
        phi = constraint_vector(q_proj, golfer_params)
        assert np.linalg.norm(phi) < 1e-4, (
            f"Constraint violation after projection: {np.linalg.norm(phi)}"
        )

    def test_idempotent(self, golfer_params: GolferParams) -> None:
        q = np.zeros(N_DOF)
        q1 = project_to_constraints(q, golfer_params)
        q2 = project_to_constraints(q1, golfer_params)
        assert np.allclose(q1, q2, atol=1e-8), "Projection should be idempotent"

    def test_raises_when_projection_does_not_converge(
        self,
        golfer_params: GolferParams,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def stuck_constraint(_q: np.ndarray, _params: GolferParams) -> np.ndarray:
            return np.ones(N_CONSTRAINTS)

        def constant_jacobian(_q: np.ndarray, _params: GolferParams) -> np.ndarray:
            return np.eye(N_CONSTRAINTS, N_DOF)

        monkeypatch.setattr(constraint_solver_module, "constraint_vector", stuck_constraint)
        monkeypatch.setattr(
            constraint_solver_module,
            "constraint_jacobian",
            constant_jacobian,
        )

        with pytest.raises(RuntimeError, match="did not converge"):
            project_to_constraints(np.zeros(N_DOF), golfer_params, max_iter=2)


class TestProjectVelocity:
    """Velocity projection must satisfy Phi_q * qdot = 0."""

    def test_zero_velocity_preserved(self, golfer_params: GolferParams) -> None:
        q = project_to_constraints(np.zeros(N_DOF), golfer_params)
        qdot = np.zeros(N_DOF)
        qdot_proj = project_velocity(q, qdot, golfer_params)
        assert np.allclose(qdot_proj, 0.0, atol=1e-10)

    def test_velocity_satisfies_constraint(self, golfer_params: GolferParams) -> None:
        q = project_to_constraints(np.zeros(N_DOF), golfer_params)
        qdot = np.ones(N_DOF) * 0.5
        qdot_proj = project_velocity(q, qdot, golfer_params)
        Phi_q = constraint_jacobian(q, golfer_params)
        violation = Phi_q @ qdot_proj
        assert np.linalg.norm(violation) < 1e-6, (
            f"Velocity constraint violation: {np.linalg.norm(violation)}"
        )


class TestConstrainedAccelerations:
    """Accelerations from KKT system must be finite and consistent."""

    def test_finite_at_rest(
        self,
        golfer_params: GolferParams,
        zero_torque: Callable[[float], tuple[float, float, float, float, float, float, float]],
    ) -> None:
        state = _make_consistent_state(golfer_params)
        qddot = constrained_accelerations(state, 0.0, golfer_params, zero_torque)
        assert qddot.shape == (N_DOF,)
        assert np.all(np.isfinite(qddot)), f"Non-finite accelerations: {qddot}"

    def test_shape(
        self,
        golfer_params: GolferParams,
        zero_torque: Callable[[float], tuple[float, float, float, float, float, float, float]],
    ) -> None:
        state = _make_consistent_state(golfer_params)
        qddot = constrained_accelerations(state, 0.0, golfer_params, zero_torque)
        assert qddot.shape == (N_DOF,)


class TestConstraintForces:
    """Lagrange multipliers must have correct shape."""

    def test_shape(
        self,
        golfer_params: GolferParams,
        zero_torque: Callable[[float], tuple[float, float, float, float, float, float, float]],
    ) -> None:
        state = _make_consistent_state(golfer_params)
        lam = constraint_forces(state, 0.0, golfer_params, zero_torque)
        assert lam.shape == (N_CONSTRAINTS,)
        assert np.all(np.isfinite(lam))


class TestEquationsOfMotion:
    """Full EOM must return proper state derivative."""

    def test_shape(
        self,
        golfer_params: GolferParams,
        zero_torque: Callable[[float], tuple[float, float, float, float, float, float, float]],
    ) -> None:
        state = _make_consistent_state(golfer_params)
        state_dot = equations_of_motion(state, 0.0, golfer_params, zero_torque)
        assert state_dot.shape == (2 * N_DOF,)
        assert np.all(np.isfinite(state_dot))

    def test_velocity_in_derivative(
        self,
        golfer_params: GolferParams,
        zero_torque: Callable[[float], tuple[float, float, float, float, float, float, float]],
    ) -> None:
        state = _make_consistent_state(golfer_params)
        state_dot = equations_of_motion(state, 0.0, golfer_params, zero_torque)
        # First N_DOF of state_dot should be qdot
        assert np.allclose(state_dot[:N_DOF], state[N_DOF:])


class TestConstraintViolation:
    """Constraint violation measure must be non-negative."""

    def test_nonneg(self, golfer_params: GolferParams) -> None:
        state = _make_consistent_state(golfer_params)
        v = constraint_violation(state, golfer_params)
        assert v >= 0.0

    def test_near_zero_after_projection(self, golfer_params: GolferParams) -> None:
        state = _make_consistent_state(golfer_params)
        v = constraint_violation(state, golfer_params)
        assert v < 1e-4, f"Violation = {v}, expected near zero"
