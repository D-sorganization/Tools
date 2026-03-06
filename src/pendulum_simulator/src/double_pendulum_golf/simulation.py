"""
Simulation engine for the driven double pendulum.

Integrates the equations of motion using scipy's solve_ivp and
stores results in a structured SimulationResult for easy access
by the GUI and analysis code.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from .physics import (
    PendulumParams,
    State,
    TorqueFunc,
    coriolis_vector,
    equations_of_motion,
    forward_kinematics,
    friction_torque_vector,
    gravity_vector,
    kinetic_energy,
    net_joint_forces,
    mass_matrix_components,
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

    Preconditions:
        - Each coefficient list has at least one element.
    Postconditions:
        - Returned function produces finite values for finite t.

    Parameters
    ----------
    coeffs_shoulder : list of float
        Polynomial coefficients [c0, c1, c2, ...] for shoulder torque.
    coeffs_wrist : list of float
        Polynomial coefficients [c0, c1, c2, ...] for wrist torque.

    Returns
    -------
    torque_func : callable (t) -> (tau1, tau2)
    """
    assert len(coeffs_shoulder) >= 1, "Need at least one coefficient for shoulder"
    assert len(coeffs_wrist) >= 1, "Need at least one coefficient for wrist"

    # numpy polyval expects highest-degree first, so reverse
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
    """Stores the complete trajectory and derived quantities.

    Attributes
    ----------
    t : np.ndarray, shape (N,)
        Time points.
    states : np.ndarray, shape (N, 4)
        State vectors [theta1, phi, dtheta1, dphi] at each time.
    params : PendulumParams
        Physical parameters used.
    torque_func : TorqueFunc
        Torque function used.
    """

    t: np.ndarray
    states: np.ndarray
    params: PendulumParams
    torque_func: TorqueFunc

    # Lazily computed caches
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
        """Get mass matrix components at time index idx."""
        assert 0 <= idx < self.n_steps, f"Index {idx} out of range [0, {self.n_steps})"
        return mass_matrix_components(self.states[idx, 1], self.params)

    def positions_at(self, idx: int) -> dict:
        """Get joint positions at time index idx."""
        assert 0 <= idx < self.n_steps
        return forward_kinematics(self.states[idx, 0], self.states[idx, 1], self.params)

    def torques_at(self, idx: int) -> Tuple[float, float]:
        """Get applied torques at time index idx."""
        assert 0 <= idx < self.n_steps
        return self.torque_func(self.t[idx])

    def accelerations_at(self, idx: int) -> np.ndarray:
        """Get angular accelerations [ddtheta1, ddphi] at time index idx."""
        assert 0 <= idx < self.n_steps
        state_dot = equations_of_motion(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )
        return state_dot[2:]

    def joint_forces_at(self, idx: int) -> dict:
        """Get net joint forces (proximal on distal) at time index idx."""
        assert 0 <= idx < self.n_steps
        qddot = self.accelerations_at(idx)
        return net_joint_forces(self.states[idx], qddot, self.params)

    def energy_at(self, idx: int) -> dict:
        """Get energy components at time index idx."""
        state = self.states[idx]
        return {
            "kinetic": kinetic_energy(state, self.params),
            "potential": potential_energy(state, self.params),
            "total": total_energy(state, self.params),
        }

    def coriolis_at(self, idx: int) -> np.ndarray:
        """Get Coriolis/centrifugal vector at time index idx."""
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return coriolis_vector(s[1], s[2], s[3], self.params)

    def gravity_at(self, idx: int) -> np.ndarray:
        """Get gravity vector at time index idx."""
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return gravity_vector(s[0], s[1], self.params)

    def friction_torques_at(self, idx: int) -> np.ndarray:
        """Get dissipative friction torque vector [tau_f1, tau_f2] at time idx.

        These are the damping + Coulomb friction torques computed from the
        joint velocities at this timestep. They are NOT part of the driving
        torque_func and represent energy removed from the system.

        Returns
        -------
        np.ndarray, shape (2,)  [N\u00b7m]   (negative = opposing motion)
        """
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return friction_torque_vector(s[2], s[3], self.params)

    def total_torques_at(self, idx: int) -> np.ndarray:
        """Get total applied torque [tau_total_1, tau_total_2] at time idx.

        Total = driving torque (from torque_func) + friction torque.
        This is the net torque that actually enters the equations of motion.

        Returns
        -------
        np.ndarray, shape (2,)  [N\u00b7m]
        """
        assert 0 <= idx < self.n_steps
        tau_drive = np.array(self.torque_func(self.t[idx]))
        tau_friction = self.friction_torques_at(idx)
        return tau_drive + tau_friction


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
) -> SimulationResult:
    """Integrate the double pendulum equations of motion.

    Preconditions:
        - initial_state has shape (4,) with finite values.
        - t_end > 0.
        - dt > 0 and dt < t_end.
    Postconditions:
        - Result contains at least 2 time points.
        - All state values in result are finite (if integration succeeded).

    Parameters
    ----------
    params : PendulumParams
    initial_state : np.ndarray, shape (4,)
    t_end : float
        Simulation duration (s).
    torque_func : callable
    dt : float
        Output time step (s). Integration uses adaptive stepping internally.
    method : str
        scipy integrator method.

    Returns
    -------
    SimulationResult
    """
    assert initial_state.shape == (
        4,
    ), f"Initial state shape must be (4,), got {initial_state.shape}"
    assert all(np.isfinite(initial_state)), "Initial state must be finite"
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    assert 0 < dt < t_end, f"dt must be in (0, t_end), got {dt}"

    t_eval = np.arange(0.0, t_end, dt)

    def ode_rhs(t, y):
        return equations_of_motion(y, t, params, torque_func)

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
        states=sol.y.T,  # transpose to (N, 4)
        params=params,
        torque_func=torque_func,
    )

    assert result.n_steps >= 2, "Simulation must produce at least 2 time points"
    return result
