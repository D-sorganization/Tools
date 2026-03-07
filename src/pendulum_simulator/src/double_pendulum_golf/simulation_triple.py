"""
Simulation engine for the driven triple pendulum.

Integrates the equations of motion using scipy's solve_ivp and
stores results in a structured TripleSimulationResult for easy access
by the GUI and analysis code.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .physics_triple import (
    TriplePendulumParams,
    State,
    TorqueFunc,
    coriolis_vector,
    equations_of_motion,
    forward_kinematics,
    friction_torque_vector,
    gravity_vector,
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
    coeffs_elbow: list[float],
    coeffs_wrist: list[float],
) -> TorqueFunc:
    """Create a torque function from polynomial coefficients.

    tau(t) = c0 + c1*t + c2*t^2 + ...
    """
    assert len(coeffs_shoulder) >= 1, "Need at least one coefficient for shoulder"
    assert len(coeffs_elbow) >= 1, "Need at least one coefficient for elbow"
    assert len(coeffs_wrist) >= 1, "Need at least one coefficient for wrist"

    p_shoulder = np.array(coeffs_shoulder[::-1])
    p_elbow = np.array(coeffs_elbow[::-1])
    p_wrist = np.array(coeffs_wrist[::-1])

    def torque_func(t: float) -> tuple[float, float, float]:
        tau1 = float(np.polyval(p_shoulder, t))
        tau2 = float(np.polyval(p_elbow, t))
        tau3 = float(np.polyval(p_wrist, t))
        return tau1, tau2, tau3

    return torque_func


# ---------------------------------------------------------------------------
# Simulation result container
# ---------------------------------------------------------------------------


@dataclass
class TripleSimulationResult:
    """Stores the complete trajectory and derived quantities."""

    t: np.ndarray
    states: np.ndarray
    params: TriplePendulumParams
    torque_func: TorqueFunc

    @property
    def n_steps(self) -> int:
        return len(self.t)

    def mass_matrix_at(self, idx: int) -> dict:
        assert 0 <= idx < self.n_steps, f"Index {idx} out of range [0, {self.n_steps})"
        s = self.states[idx]
        return mass_matrix_components(s[1], s[2], self.params)

    def positions_at(self, idx: int) -> dict:
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return forward_kinematics(s[0], s[1], s[2], self.params)

    def torques_at(self, idx: int) -> tuple[float, float, float]:
        assert 0 <= idx < self.n_steps
        return self.torque_func(self.t[idx])

    def accelerations_at(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.n_steps
        state_dot = equations_of_motion(
            self.states[idx], self.t[idx], self.params, self.torque_func
        )
        return state_dot[3:]

    def joint_forces_at(self, idx: int) -> dict:
        assert 0 <= idx < self.n_steps
        qddot = self.accelerations_at(idx)
        return net_joint_forces(self.states[idx], qddot, self.params)

    def coriolis_at(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return coriolis_vector(s[1], s[2], s[3], s[4], s[5], self.params)

    def gravity_at(self, idx: int) -> np.ndarray:
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return gravity_vector(s[0], s[1], s[2], self.params)

    def energy_at(self, idx: int) -> dict:
        state = self.states[idx]
        return {
            "kinetic": kinetic_energy(state, self.params),
            "potential": potential_energy(state, self.params),
            "total": total_energy(state, self.params),
        }

    def friction_torques_at(self, idx: int) -> np.ndarray:
        """Get dissipative friction torque vector at time idx.

        Returns
        -------
        np.ndarray, shape (3,)  [N·m]
        """
        assert 0 <= idx < self.n_steps
        s = self.states[idx]
        return friction_torque_vector(s[3], s[4], s[5], self.params)

    def total_torques_at(self, idx: int) -> np.ndarray:
        """Get total applied torque at time idx.

        Total = driving torque + friction torque.

        Returns
        -------
        np.ndarray, shape (3,)  [N·m]
        """
        assert 0 <= idx < self.n_steps
        tau_drive = np.array(self.torque_func(self.t[idx]))
        tau_friction = self.friction_torques_at(idx)
        return np.asarray(tau_drive + tau_friction)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------


def run_simulation(
    params: TriplePendulumParams,
    initial_state: State,
    t_end: float,
    torque_func: TorqueFunc,
    dt: float = 0.005,
    method: str = "DOP853",
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> TripleSimulationResult:
    """Integrate the triple pendulum equations of motion.

    Performance notes
    -----------------
    - ``DOP853`` is an 8th-order Runge-Kutta method that is fast for
      non-stiff problems while being accurate enough for chaotic systems.
    - ``max_step`` is intentionally NOT set here — letting the solver
      choose an adaptive step is ~10-50x faster than forcing ``dt`` as
      the maximum step size.
    - The output is resampled to a uniform ``t_eval`` grid at spacing ``dt``
      after integration, giving the GUI a predictable frame count without
      slowing down the solver.
    - Tolerances: ``rtol=1e-6`` / ``atol=1e-8`` are appropriate for
      visualisation-quality results.  Use tighter values only when
      quantitative energy conservation is required.
    """
    assert initial_state.shape == (
        6,
    ), f"Initial state shape must be (6,), got {initial_state.shape}"
    assert all(np.isfinite(initial_state)), "Initial state must be finite"
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    assert 0 < dt < t_end, f"dt must be in (0, t_end), got {dt}"

    # Uniform output grid — solver adapts its internal step freely
    t_eval = np.arange(0.0, t_end, dt)

    def ode_rhs(t: float, y: np.ndarray) -> np.ndarray:
        return equations_of_motion(y, t, params, torque_func)

    sol = solve_ivp(
        ode_rhs,
        t_span=(0.0, t_end),
        y0=initial_state,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        # max_step deliberately omitted — adaptive step is much faster
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    result = TripleSimulationResult(
        t=sol.t,
        states=sol.y.T,
        params=params,
        torque_func=torque_func,
    )

    assert result.n_steps >= 2, "Simulation must produce at least 2 time points"
    return result
