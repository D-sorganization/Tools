"""
Simulation engine for the golfer upper-body model.

Integrates the constrained equations of motion and stores results
in a GolferSimulationResult for GUI and analysis access.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

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

# ---------------------------------------------------------------------------
# Polynomial torque builder
# ---------------------------------------------------------------------------


def make_polynomial_torque(
    coeffs_hub: list[float],
    coeffs_rs: list[float],
    coeffs_re: list[float],
    coeffs_rh: list[float],
    coeffs_ls: list[float],
    coeffs_le: list[float],
    coeffs_lh: list[float],
) -> TorqueFunc:
    """Create a torque function from polynomial coefficients for each joint.

    tau_i(t) = c0 + c1*t + c2*t^2 + ...
    """
    polys = []
    for name, coeffs in [
        ("hub", coeffs_hub),
        ("rs", coeffs_rs),
        ("re", coeffs_re),
        ("rh", coeffs_rh),
        ("ls", coeffs_ls),
        ("le", coeffs_le),
        ("lh", coeffs_lh),
    ]:
        assert len(coeffs) >= 1, f"Need at least one coefficient for {name}"
        polys.append(np.array(coeffs[::-1]))

    def torque_func(
        t: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        return tuple(float(np.polyval(p, t)) for p in polys)  # type: ignore[return-value]

    return torque_func


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class GolferSimulationResult:
    """Stores the complete trajectory and provides derived-quantity accessors."""

    t: np.ndarray  # shape (n_steps,)
    states: np.ndarray  # shape (n_steps, 16)
    params: GolferParams
    torque_func: TorqueFunc

    @property
    def n_steps(self) -> int:
        return len(self.t)

    def q_at(self, idx: int) -> np.ndarray:
        """Generalized coordinates at time index."""
        assert 0 <= idx < self.n_steps
        return self.states[idx, :N_DOF]

    def qdot_at(self, idx: int) -> np.ndarray:
        """Generalized velocities at time index."""
        assert 0 <= idx < self.n_steps
        return self.states[idx, N_DOF:]

    def mass_matrix_at(self, idx: int) -> np.ndarray:
        """8×8 mass matrix at time index."""
        assert 0 <= idx < self.n_steps
        return mass_matrix(self.q_at(idx), self.params)

    def positions_at(self, idx: int) -> dict:
        """Forward kinematics at time index."""
        assert 0 <= idx < self.n_steps
        return forward_kinematics(self.q_at(idx), self.params)

    def torques_at(self, idx: int) -> tuple[float, float, float, float, float, float, float]:
        """Applied driving torques at time index."""
        assert 0 <= idx < self.n_steps
        return self.torque_func(self.t[idx])

    def accelerations_at(self, idx: int) -> np.ndarray:
        """Joint accelerations at time index."""
        assert 0 <= idx < self.n_steps
        return constrained_accelerations(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )

    def joint_forces_at(self, idx: int) -> dict:
        """Net joint forces at time index."""
        assert 0 <= idx < self.n_steps
        q = self.q_at(idx)
        qdot = self.qdot_at(idx)
        qddot = self.accelerations_at(idx)
        return net_joint_forces(q, qdot, qddot, self.params)

    def constraint_forces_at(self, idx: int) -> np.ndarray:
        """Lagrange multiplier (constraint) forces at time index."""
        assert 0 <= idx < self.n_steps
        return constraint_forces(self.states[idx], self.t[idx], self.params, self.torque_func)

    def constraint_violation_at(self, idx: int) -> float:
        """Constraint violation magnitude at time index."""
        assert 0 <= idx < self.n_steps
        return constraint_violation(self.states[idx], self.params)

    def coriolis_at(self, idx: int) -> np.ndarray:
        """Coriolis/centrifugal torques at time index."""
        assert 0 <= idx < self.n_steps
        return coriolis_matrix(self.q_at(idx), self.qdot_at(idx), self.params)

    def gravity_at(self, idx: int) -> np.ndarray:
        """Gravitational torques at time index."""
        assert 0 <= idx < self.n_steps
        return gravity_vector(self.q_at(idx), self.params)

    def energy_at(self, idx: int) -> dict:
        """Energy decomposition at time index."""
        state = self.states[idx]
        return {
            "kinetic": kinetic_energy(state[:N_DOF], state[N_DOF:], self.params),
            "potential": potential_energy(state, self.params),
            "total": total_energy(state, self.params),
        }

    def friction_torques_at(self, idx: int) -> np.ndarray:
        """Friction torques at time index."""
        assert 0 <= idx < self.n_steps
        return friction_torque_vector(self.qdot_at(idx), self.params)

    def total_torques_at(self, idx: int) -> np.ndarray:
        """Total applied torque (drive + friction) at time index."""
        assert 0 <= idx < self.n_steps
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

    Returns
    -------
    GolferSimulationResult
    """
    assert initial_state.shape == (2 * N_DOF,), (
        f"Initial state shape must be ({2 * N_DOF},), got {initial_state.shape}"
    )
    assert np.all(np.isfinite(initial_state)), "Initial state must be finite"
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    assert 0 < dt < t_end, f"dt must be in (0, t_end), got {dt}"

    # Project initial conditions onto constraint manifold
    q0 = project_to_constraints(initial_state[:N_DOF], params)
    qdot0 = project_velocity(q0, initial_state[N_DOF:], params)
    y0 = np.concatenate([q0, qdot0])

    t_eval = np.arange(0.0, t_end, dt)

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        return equations_of_motion(y, t, params, torque_func, alpha, beta)

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
