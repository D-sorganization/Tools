"""
Simulation engine for the driven double pendulum golf model.

Integrates the equations of motion using scipy's solve_ivp and
stores results in a structured SimulationResult for easy access
by the GUI and analysis code.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .physics import (
    JointLimits,
    PendulumParams,
    State,
    TorqueClamp,
    TorqueFunc,
    base_force,
    clamp_torque,
    control_vector,
    coriolis_vector,
    equations_of_motion,
    forward_kinematics,
    friction_torque_vector,
    gravity_vector,
    joint_velocities,
    kinetic_energy,
    mass_matrix_components,
    net_joint_forces,
    potential_energy,
    total_energy,
)

# ---------------------------------------------------------------------------
# Polynomial torque builder
# ---------------------------------------------------------------------------


def make_polynomial_torque(
    coeffs_shoulder: list[float],
    coeffs_wrist: list[float],
) -> TorqueFunc:
    """Create a torque function from polynomial coefficients.

    tau(t) = c0 + c1*t + c2*t^2 + ...

    Pre: each list has >= 1 element.
    Post: returned function produces finite values for finite t.
    """
    assert len(coeffs_shoulder) >= 1, "Need at least one coefficient for shoulder"
    assert len(coeffs_wrist) >= 1, "Need at least one coefficient for wrist"

    p_shoulder = np.array(coeffs_shoulder[::-1])
    p_wrist = np.array(coeffs_wrist[::-1])

    def torque_func(t: float) -> Tuple[float, float]:
        tau1 = float(np.polyval(p_shoulder, t))
        tau2 = float(np.polyval(p_wrist, t))
        return tau1, tau2

    return torque_func


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Stores the complete trajectory and derived quantities."""

    t: np.ndarray
    states: np.ndarray
    params: PendulumParams
    torque_func: TorqueFunc
    limits: Optional[JointLimits] = None
    clamp: Optional[TorqueClamp] = None

    _mass_matrices: Optional[np.ndarray] = field(default=None, repr=False)
    _positions: Optional[list] = field(default=None, repr=False)

    @property
    def n_steps(self) -> int:
        return len(self.t)

    @property
    def theta1(self) -> np.ndarray:
        return self.states[:, 0]

    @property
    def phi(self) -> np.ndarray:
        return self.states[:, 1]

    @property
    def dtheta1(self) -> np.ndarray:
        return self.states[:, 2]

    @property
    def dphi(self) -> np.ndarray:
        return self.states[:, 3]

    def mass_matrix_at(self, idx: int) -> dict:
        assert 0 <= idx < self.n_steps
        return mass_matrix_components(self.states[idx, 1], self.params)

    def positions_at(self, idx: int) -> dict:
        assert 0 <= idx < self.n_steps
        return forward_kinematics(self.states[idx, 0], self.states[idx, 1], self.params)

    def torques_at(self, idx: int) -> Tuple[float, float]:
        assert 0 <= idx < self.n_steps
        return self.torque_func(self.t[idx])

    def accelerations_at(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.n_steps
        state_dot = equations_of_motion(
            self.states[idx],
            self.t[idx],
            self.params,
            self.torque_func,
            self.limits,
            self.clamp,
        )
        return state_dot[2:]

    def joint_forces_at(self, idx: int) -> dict:
        assert 0 <= idx < self.n_steps
        qddot = self.accelerations_at(idx)
        return net_joint_forces(self.states[idx], qddot, self.params)

    def joint_velocities_at(self, idx: int) -> dict:
        """Get linear joint velocities at time index idx."""
        assert 0 <= idx < self.n_steps
        return joint_velocities(self.states[idx], self.params)

    def base_force_at(self, idx: int) -> dict:
        """Get base reaction force at time index idx."""
        assert 0 <= idx < self.n_steps
        qddot = self.accelerations_at(idx)
        return base_force(self.states[idx], qddot, self.params)

    def control_vector_at(self, idx: int) -> dict:
        """Get control vector at time index idx."""
        assert 0 <= idx < self.n_steps
        qddot = self.accelerations_at(idx)
        return control_vector(self.states[idx], qddot, self.params, self.limits)

    def energy_at(self, idx: int) -> dict:
        state = self.states[idx]
        return {
            "kinetic": kinetic_energy(state, self.params),
            "potential": potential_energy(state, self.params),
            "total": total_energy(state, self.params),
        }

    def coriolis_at(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return coriolis_vector(s[1], s[2], s[3], self.params)

    def gravity_at(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return gravity_vector(s[0], s[1], self.params)

    def friction_torques_at(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return friction_torque_vector(s[2], s[3], self.params)

    def total_torques_at(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.n_steps
        tau_drive = np.array(self.torque_func(self.t[idx]))
        if self.clamp is not None:
            tau_drive = clamp_torque(tau_drive, self.clamp)
        tau_friction = self.friction_torques_at(idx)
        return np.asarray(tau_drive + tau_friction)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_simulation(
    params: PendulumParams,
    initial_state: State,
    t_end: float,
    torque_func: TorqueFunc,
    dt: float = 0.005,
    method: str = "RK45",
    limits: Optional[JointLimits] = None,
    clamp: Optional[TorqueClamp] = None,
) -> SimulationResult:
    """Integrate the double pendulum equations of motion.

    Pre: initial_state shape (4,), finite. t_end > 0. dt in (0, t_end).
    Post: result has >= 2 time points, all finite.
    """
    assert initial_state.shape == (4,)
    assert all(np.isfinite(initial_state))
    assert t_end > 0 and 0 < dt < t_end

    t_eval = np.arange(0.0, t_end, dt)

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        return equations_of_motion(y, t, params, torque_func, limits, clamp)

    sol = solve_ivp(
        ode_rhs,
        t_span=(0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        method=method,
        rtol=1e-8,
        atol=1e-10,
        max_step=dt,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    result = SimulationResult(
        t=sol.t,
        states=sol.y.T,
        params=params,
        torque_func=torque_func,
        limits=limits,
        clamp=clamp,
    )

    assert result.n_steps >= 2
    return result
