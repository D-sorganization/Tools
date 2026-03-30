from typing import Any, TypedDict

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore

from .config import PhysicsConfig


class VectorsDict(TypedDict):
    """Container for vectors."""

    total: tuple[np.ndarray, np.ndarray]
    centrifugal: tuple[np.ndarray, np.ndarray]
    coriolis: tuple[np.ndarray, np.ndarray]


class PhysicsEngine:
    """Core domain logic for Lagrangian mechanics."""

    def __init__(self, config: PhysicsConfig) -> None:
        """DbC: Assume non-null config."""
        assert config is not None, "Config cannot be None"
        self.cfg = config

    def equations_of_motion(self, t: float, state: list[float]) -> list[float]:
        """Calculates instantaneous accelerations using Augmented Lagrangian."""
        assert len(state) == 4, "State vector must be length 4."
        theta1, omega1, theta2, omega2 = state
        delta = theta1 - theta2

        # Base Conservative Lagrangian Free Dynamics
        den1 = self.cfg.l1 * (
            2 * self.cfg.m1 + self.cfg.m2 - self.cfg.m2 * np.cos(2 * delta)
        )
        if den1 < 1e-12:
            den1 = 1e-12  # DbC fail-safe

        alpha1_free = (
            -self.cfg.gravity * (2 * self.cfg.m1 + self.cfg.m2) * np.sin(theta1)
            - self.cfg.m2 * self.cfg.gravity * np.sin(theta1 - 2 * theta2)
            - 2
            * np.sin(delta)
            * self.cfg.m2
            * (omega2**2 * self.cfg.l2 + omega1**2 * self.cfg.l1 * np.cos(delta))
        ) / den1

        den2 = self.cfg.l2 * (
            2 * self.cfg.m1 + self.cfg.m2 - self.cfg.m2 * np.cos(2 * delta)
        )
        if den2 < 1e-12:
            den2 = 1e-12  # DbC fail-safe

        alpha2_free = (
            2
            * np.sin(delta)
            * (
                omega1**2 * self.cfg.l1 * (self.cfg.m1 + self.cfg.m2)
                + self.cfg.gravity * (self.cfg.m1 + self.cfg.m2) * np.cos(theta1)
                + omega2**2 * self.cfg.l2 * self.cfg.m2 * np.cos(delta)
            )
        ) / den2

        # Generalized Non-Conservative Forces (Damping + Driven Oscillating Torque)
        Q1 = -self.cfg.damp1 * omega1 + self.cfg.amp1 * np.sin(self.cfg.freq1 * t)
        Q2 = -self.cfg.damp2 * omega2 + self.cfg.amp2 * np.sin(self.cfg.freq2 * t)

        # Augment Free Accelerations with the Mass Matrix M^{-1} * Q projection
        det = (
            self.cfg.m2
            * self.cfg.l1**2
            * self.cfg.l2**2
            * (self.cfg.m1 + self.cfg.m2 * np.sin(delta) ** 2)
        )
        if det > 1e-12:
            M22 = self.cfg.m2 * self.cfg.l2**2
            M11 = (self.cfg.m1 + self.cfg.m2) * self.cfg.l1**2
            M12 = self.cfg.m2 * self.cfg.l1 * self.cfg.l2 * np.cos(delta)

            alpha1_q = (M22 * Q1 - M12 * Q2) / det
            alpha2_q = (-M12 * Q1 + M11 * Q2) / det
        else:
            alpha1_q, alpha2_q = 0.0, 0.0

        alpha1 = alpha1_free + alpha1_q
        alpha2 = alpha2_free + alpha2_q

        return [omega1, alpha1, omega2, alpha2]

    def solve(self, duration: float, dt: float) -> dict[str, Any]:
        """Integrates physics over specified duration."""
        assert duration > 0 and dt > 0, "Duration and dt must be positive."
        t_eval = np.arange(0, duration, dt)
        initial_state = [
            self.cfg.theta1,
            self.cfg.omega1,
            self.cfg.theta2,
            self.cfg.omega2,
        ]

        res = solve_ivp(
            fun=self.equations_of_motion,
            t_span=[0, duration],
            y0=initial_state,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-8,
        )
        if not res.success:
            raise RuntimeError(f"ODE integration failed: {res.message}")

        return self._extract_physics(res.y, t_eval)

    def _extract_physics(self, y: np.ndarray, t_eval: np.ndarray) -> dict[str, Any]:
        """Convert angular array into all Cartesian force vectors."""
        assert y.shape[0] == 4, "Input y array must have 4 rows."

        theta1, omega1, theta2, omega2 = y[0, :], y[1, :], y[2, :], y[3, :]

        alpha1 = np.zeros_like(theta1)
        alpha2 = np.zeros_like(theta2)
        for i in range(len(theta1)):
            derivs = self.equations_of_motion(
                t_eval[i], [theta1[i], omega1[i], theta2[i], omega2[i]]
            )
            alpha1[i], alpha2[i] = derivs[1], derivs[3]

        # Kinematics
        x1 = self.cfg.l1 * np.sin(theta1)
        y1 = -self.cfg.l1 * np.cos(theta1)
        x2 = x1 + self.cfg.l2 * np.sin(theta2)
        y2 = y1 - self.cfg.l2 * np.cos(theta2)

        # Unit Vectors (n pointing outward along rod, t pointing tangentially)
        # e_r1 points out along rod 1. e_t1 orthogonal.
        e_r1_x, e_r1_y = np.sin(theta1), -np.cos(theta1)
        e_t1_x, e_t1_y = np.cos(theta1), np.sin(theta1)

        e_r2_x, e_r2_y = np.sin(theta2), -np.cos(theta2)
        e_t2_x, e_t2_y = np.cos(theta2), np.sin(theta2)

        # Cartesian Accelerations
        a1_x = self.cfg.l1 * (alpha1 * e_t1_x - omega1**2 * e_r1_x)
        a1_y = self.cfg.l1 * (alpha1 * e_t1_y - omega1**2 * e_r1_y)

        a2_x = a1_x + self.cfg.l2 * (alpha2 * e_t2_x - omega2**2 * e_r2_x)
        a2_y = a1_y + self.cfg.l2 * (alpha2 * e_t2_y - omega2**2 * e_r2_y)

        # Net Forces
        F1_t = (self.cfg.m1 * a1_x, self.cfg.m1 * a1_y)
        F2_t = (self.cfg.m2 * a2_x, self.cfg.m2 * a2_y)

        # CF Force: mass * dot(theta)^2 * r. Note: outward is e_r.
        # Node 1: CF force is purely due to its own rotation.
        cf1_x = self.cfg.m1 * self.cfg.l1 * omega1**2 * e_r1_x
        cf1_y = self.cfg.m1 * self.cfg.l1 * omega1**2 * e_r1_y

        # Node 2: CF from rotating frame 1 + its own rotation
        cf2_x = self.cfg.m2 * (
            self.cfg.l1 * omega1**2 * e_r1_x + self.cfg.l2 * omega2**2 * e_r2_x
        )
        cf2_y = self.cfg.m2 * (
            self.cfg.l1 * omega1**2 * e_r1_y + self.cfg.l2 * omega2**2 * e_r2_y
        )

        # Coriolis Force in Cartesian (on Node 2 from Node 1 frame):
        # 2 * m2 * (omega1 k x v_rel) = 2 * m2 * l2 * omega1 * (omega2 - omega1) * e_r2
        cor2_mag = 2 * self.cfg.m2 * self.cfg.l2 * omega1 * (omega2 - omega1)
        cor2_x = cor2_mag * e_r2_x
        cor2_y = cor2_mag * e_r2_y

        # Coriolis on Node 1 is purely zero in Cartesian root frame.
        cor1_x = np.zeros_like(cor2_x)
        cor1_y = np.zeros_like(cor2_y)

        # Nodal Torques (for graphing)
        tau1 = self.cfg.m1 * self.cfg.l1**2 * alpha1
        tau2 = self.cfg.m2 * self.cfg.l2**2 * alpha2

        pos = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        v1: VectorsDict = {
            "total": F1_t,
            "centrifugal": (cf1_x, cf1_y),
            "coriolis": (cor1_x, cor1_y),
        }
        v2: VectorsDict = {
            "total": F2_t,
            "centrifugal": (cf2_x, cf2_y),
            "coriolis": (cor2_x, cor2_y),
        }
        return {
            "pos": pos,
            "t_eval": t_eval,
            "v1": v1,
            "v2": v2,
            "tau1": tau1,
            "tau2": tau2,
        }
