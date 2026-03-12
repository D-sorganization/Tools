"""
Simulation engine for the golfer upper-body model.

Integrates the constrained equations of motion and stores results
in a GolferSimulationResult for GUI and analysis access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from .physics import JointLimitsNDOF

from .constraint_solver import (
    constraint_forces,
    constraint_violation,
    constrained_accelerations,
    equations_of_motion,
    project_to_constraints,
    project_velocity,
)
from .physics_golfer import (
    N_DOF,
    GolferParams,
    State,
    TorqueFunc,
    coriolis_matrix,
    forward_kinematics,
    friction_torque_vector,
    gravity_vector,
    kinetic_energy,
    mass_matrix,
    net_joint_forces,
    potential_energy,
    total_energy,
)
from .simulation_result_base import TrajectoryResultMixin

# Re-export from shared utility for backwards compatibility (DRY — #1041)
from .torque_utils import make_polynomial_torque  # noqa: F401

# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class GolferSimulationResult(TrajectoryResultMixin):
    """Stores the complete trajectory and provides derived-quantity accessors."""

    t: np.ndarray  # shape (n_steps,)
    states: np.ndarray  # shape (n_steps, 16)
    params: GolferParams
    torque_func: TorqueFunc

    def __post_init__(self) -> None:
        self._validate_trajectory(expected_state_width=2 * N_DOF)

    def q_at(self, idx: int) -> np.ndarray:
        """Generalized coordinates at time index."""
        self._check_idx(idx)
        return self.states[idx, :N_DOF]

    def qdot_at(self, idx: int) -> np.ndarray:
        """Generalized velocities at time index."""
        self._check_idx(idx)
        return self.states[idx, N_DOF:]

    def mass_matrix_at(self, idx: int) -> np.ndarray:
        """8×8 mass matrix at time index."""
        self._check_idx(idx)
        return mass_matrix(self.q_at(idx), self.params)

    def positions_at(self, idx: int) -> dict:
        """Forward kinematics at time index."""
        self._check_idx(idx)
        return forward_kinematics(self.q_at(idx), self.params)

    def torques_at(
        self, idx: int
    ) -> tuple[float, float, float, float, float, float, float]:
        """Applied driving torques at time index."""
        self._check_idx(idx)
        return self.torque_func(self.t[idx])

    def accelerations_at(self, idx: int) -> np.ndarray:
        """Joint accelerations at time index."""
        self._check_idx(idx)
        return constrained_accelerations(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )

    def joint_forces_at(self, idx: int) -> dict:
        """Net joint forces at time index."""
        self._check_idx(idx)
        q = self.q_at(idx)
        qdot = self.qdot_at(idx)
        qddot = self.accelerations_at(idx)
        return net_joint_forces(q, qdot, qddot, self.params)

    def constraint_forces_at(self, idx: int) -> np.ndarray:
        """Lagrange multiplier (constraint) forces at time index."""
        self._check_idx(idx)
        return constraint_forces(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )

    def constraint_violation_at(self, idx: int) -> float:
        """Constraint violation magnitude at time index."""
        self._check_idx(idx)
        return constraint_violation(self.states[idx], self.params)

    def coriolis_at(self, idx: int) -> np.ndarray:
        """Coriolis/centrifugal torques at time index."""
        self._check_idx(idx)
        return coriolis_matrix(self.q_at(idx), self.qdot_at(idx), self.params)

    def gravity_at(self, idx: int) -> np.ndarray:
        """Gravitational torques at time index."""
        self._check_idx(idx)
        return gravity_vector(self.q_at(idx), self.params)

    def energy_at(self, idx: int) -> dict:
        """Energy decomposition at time index."""
        self._check_idx(idx)
        state = self.states[idx]
        return {
            "kinetic": kinetic_energy(state[:N_DOF], state[N_DOF:], self.params),
            "potential": potential_energy(state, self.params),
            "total": total_energy(state, self.params),
        }

    def friction_torques_at(self, idx: int) -> np.ndarray:
        """Friction torques at time index."""
        self._check_idx(idx)
        return friction_torque_vector(self.qdot_at(idx), self.params)

    def total_torques_at(self, idx: int) -> np.ndarray:
        """Total applied torque (drive + friction) at time index."""
        self._check_idx(idx)
        tau_drive = np.zeros(N_DOF)
        tau_drive[:7] = self.torque_func(self.t[idx])
        tau_friction = self.friction_torques_at(idx)
        result: np.ndarray = tau_drive + tau_friction
        return result


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_simulation(
    params: GolferParams,
    initial_state: State,
    t_end: float,
    torque_func: TorqueFunc,
    dt: float = 0.005,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    alpha: float = 5.0,
    beta: float = 5.0,
    torque_limits: np.ndarray | None = None,
    limits: "JointLimitsNDOF | None" = None,
    clamp: np.ndarray | None = None,
) -> GolferSimulationResult:
    """Integrate the constrained golfer equations of motion.

    Parameters
    ----------
    params : GolferParams
    initial_state : np.ndarray, shape (16,)
    t_end : float
    torque_func : callable
    dt : float — output time step
    method : str — ODE solver method
    rtol, atol : float — solver tolerances
    alpha, beta : float — Baumgarte stabilization gains
    torque_limits : np.ndarray, shape (7,), optional
        Per-joint torque saturation limits.
    limits : JointLimitsNDOF, optional
        Joint angle limits with penalty stiffness.

    Returns
    -------
    GolferSimulationResult
    """
    assert initial_state.shape == (
        2 * N_DOF,
    ), f"Initial state shape must be ({2 * N_DOF},), got {initial_state.shape}"
    assert np.all(np.isfinite(initial_state)), "Initial state must be finite"
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    assert 0 < dt < t_end, f"dt must be in (0, t_end), got {dt}"

    # Merge clamp kwarg (from SimulationPanel) with torque_limits (legacy)
    effective_torque_limits = torque_limits if torque_limits is not None else clamp

    # Project initial conditions onto constraint manifold
    q0 = project_to_constraints(initial_state[:N_DOF], params)
    qdot0 = project_velocity(q0, initial_state[N_DOF:], params)
    y0 = np.concatenate([q0, qdot0])

    t_eval = np.arange(0.0, t_end, dt)

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        dydt = equations_of_motion(
            y, t, params, torque_func, alpha, beta, effective_torque_limits
        )
        # Apply joint limit penalty torques if enabled
        if limits is not None:
            from .physics import joint_limit_torque_ndof

            q = y[:N_DOF]
            qdot = y[N_DOF:]
            # Only apply limits to the 7 actuated DOFs (not club theta)
            q_actuated = q[:7]
            qdot_actuated = qdot[:7]
            tau_limit = joint_limit_torque_ndof(q_actuated, qdot_actuated, limits)
            # Add penalty via M^-1 * tau_limit (full 8-DOF mass matrix)
            M = mass_matrix(q, params)
            tau_full = np.zeros(N_DOF)
            tau_full[:7] = tau_limit
            qddot_correction = np.linalg.solve(M, tau_full)
            dydt[N_DOF:] += qddot_correction
        return dydt

    sol = solve_ivp(
        ode_rhs,
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    result = GolferSimulationResult(
        t=sol.t,
        states=sol.y.T,
        params=params,
        torque_func=torque_func,
    )

    assert result.n_steps >= 2, "Simulation must produce at least 2 time points"
    return result
