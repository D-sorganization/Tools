"""Native physics wrapper for pendulum simulator.

This module attempts to use the compiled Rust physics kernel via FFI when available.
If the native library is not compiled or accessible, it falls back to pure-Python
NumPy implementations.

Usage:
    from physics_native import DoublePendulum, TriplePendulum, Golfer

    # Automatically uses Rust if available, falls back to NumPy
    model = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0)
    M = model.mass_matrix(q)
"""

import logging
from typing import Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import the compiled Rust module
HAS_NATIVE = False
try:
    import pendulum_core

    HAS_NATIVE = True
    NATIVE_ERROR = None
except ImportError as e:
    NATIVE_ERROR = str(e)
    logger.debug("Rust pendulum_core unavailable: %s", e)


class DoublePendulumParams:
    """Parameters for the 2-DOF double pendulum model."""

    def __init__(
        self,
        m1: float,
        m2: float,
        l1: float,
        l2: float,
        g: float = 9.81,
        friction1: float = 0.0,
        friction2: float = 0.0,
        m_clubhead: float = 0.0,
    ):
        """Initialize double pendulum parameters.

        Args:
            m1: Mass of arm segment (kg)
            m2: Mass of club segment (kg)
            l1: Length of arm segment (m)
            l2: Length of club segment (m)
            g: Gravitational acceleration (m/s²)
            friction1: Friction coefficient for first joint
            friction2: Friction coefficient for second joint
            m_clubhead: Clubhead point mass at the tip (kg)
        """
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.friction1 = friction1
        self.friction2 = friction2
        self.m_clubhead = m_clubhead

    def to_rust(self):
        """Convert to Rust parameter object (if native available)."""
        if HAS_NATIVE:
            return pendulum_core.PyDoublePendulumParams(
                self.m1,
                self.m2,
                self.l1,
                self.l2,
                self.g,
                self.friction1,
                self.friction2,
                self.m_clubhead,
            )
        return self


class DoublePendulum:
    """Double pendulum physics model (2-DOF)."""

    def __init__(
        self,
        m1: float,
        m2: float,
        l1: float,
        l2: float,
        g: float = 9.81,
        m_clubhead: float = 0.0,
    ):
        self.params = DoublePendulumParams(
            m1=m1, m2=m2, l1=l1, l2=l2, g=g, m_clubhead=m_clubhead
        )
        self.use_native = HAS_NATIVE

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute the 2x2 mass matrix M(q)."""
        if self.use_native:
            try:
                result = pendulum_core.py_double_mass_matrix(q.tolist(), self.params.to_rust())
                return np.array(result, dtype=np.float64)
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    "Rust mass_matrix call failed (%s), falling back to NumPy: %s",
                    type(e).__name__,
                    e,
                )

        # NumPy fallback
        phi = q[1]
        cos_phi = np.cos(phi)
        me = self.params.m2 + self.params.m_clubhead
        m00 = (self.params.m1 + me) * self.params.l1**2 + me * self.params.l2**2
        m00 += 2.0 * me * self.params.l1 * self.params.l2 * cos_phi
        m01 = me * self.params.l2**2 + me * self.params.l1 * self.params.l2 * cos_phi
        m11 = me * self.params.l2**2
        return np.array([[m00, m01], [m01, m11]], dtype=np.float64)

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Compute the gravity vector G(q)."""
        if self.use_native:
            try:
                result = pendulum_core.py_double_gravity_vector(
                    q.tolist(), self.params.to_rust()
                )
                return np.array(result, dtype=np.float64)
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    "Rust gravity_vector call failed (%s), falling back to NumPy",
                    type(e).__name__,
                )

        # NumPy fallback
        theta1 = q[0]
        theta2 = theta1 + q[1]
        me = self.params.m2 + self.params.m_clubhead
        g0 = (self.params.m1 + me) * self.params.g * self.params.l1 * np.sin(
            theta1
        ) + me * self.params.g * self.params.l2 * np.sin(theta2)
        g1 = me * self.params.g * self.params.l2 * np.sin(theta2)
        return np.array([g0, g1], dtype=np.float64)

    def coriolis(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Compute the Coriolis vector C(q, qdot)."""
        if self.use_native:
            try:
                result = pendulum_core.py_double_coriolis(
                    q.tolist(), qdot.tolist(), self.params.to_rust()
                )
                return np.array(result, dtype=np.float64)
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    "Rust coriolis call failed (%s), falling back to NumPy",
                    type(e).__name__,
                )

        # NumPy fallback
        phi = q[1]
        me = self.params.m2 + self.params.m_clubhead
        h = -me * self.params.l1 * self.params.l2 * np.sin(phi)
        c0 = h * (2.0 * qdot[0] * qdot[1] + qdot[1] ** 2)
        c1 = -h * qdot[0] ** 2
        return np.array([c0, c1], dtype=np.float64)

    def forward_kinematics(self, q: np.ndarray) -> Dict[str, float]:
        """Compute forward kinematics."""
        if self.use_native:
            try:
                return pendulum_core.py_double_forward_kinematics(
                    q.tolist(), self.params.to_rust()
                )
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    "Rust forward_kinematics call failed (%s), falling back to NumPy",
                    type(e).__name__,
                )

        # NumPy fallback
        theta1 = q[0]
        theta2 = theta1 + q[1]
        wrist_x = self.params.l1 * np.sin(theta1)
        wrist_y = -self.params.l1 * np.cos(theta1)
        club_tip_x = wrist_x + self.params.l2 * np.sin(theta2)
        club_tip_y = wrist_y - self.params.l2 * np.cos(theta2)
        return {
            "wrist_x": float(wrist_x),
            "wrist_y": float(wrist_y),
            "club_tip_x": float(club_tip_x),
            "club_tip_y": float(club_tip_y),
            "theta1": float(theta1),
            "theta2": float(theta2),
        }


class GolferParams:
    """Parameters for the 8-DOF golfer model."""

    def __init__(
        self,
        l_hub: float,
        m_hub: float,
        d_rs: float,
        d_ls: float,
        l_r_upper: float,
        m_r_upper: float,
        l_r_fore: float,
        m_r_fore: float,
        l_l_upper: float,
        m_l_upper: float,
        l_l_fore: float,
        m_l_fore: float,
        l_club: float,
        m_club: float,
        m_clubhead: float,
        grip_right: float,
        grip_left: float,
        g: float = 9.81,
    ):
        """Initialize golfer model parameters."""
        self.l_hub = l_hub
        self.m_hub = m_hub
        self.d_rs = d_rs
        self.d_ls = d_ls
        self.l_r_upper = l_r_upper
        self.m_r_upper = m_r_upper
        self.l_r_fore = l_r_fore
        self.m_r_fore = m_r_fore
        self.l_l_upper = l_l_upper
        self.m_l_upper = m_l_upper
        self.l_l_fore = l_l_fore
        self.m_l_fore = m_l_fore
        self.l_club = l_club
        self.m_club = m_club
        self.m_clubhead = m_clubhead
        self.grip_right = grip_right
        self.grip_left = grip_left
        self.g = g

    def to_rust(self):
        """Convert to Rust parameter object (if native available)."""
        if HAS_NATIVE:
            return pendulum_core.PyGolferParams(
                self.l_hub,
                self.m_hub,
                self.d_rs,
                self.d_ls,
                self.l_r_upper,
                self.m_r_upper,
                self.l_r_fore,
                self.m_r_fore,
                self.l_l_upper,
                self.m_l_upper,
                self.l_l_fore,
                self.m_l_fore,
                self.l_club,
                self.m_club,
                self.m_clubhead,
                self.grip_right,
                self.grip_left,
                self.g,
            )
        return self


class Golfer:
    """Golfer upper body physics model (8-DOF with 4 constraints)."""

    def __init__(
        self,
        l_hub: float,
        m_hub: float,
        d_rs: float,
        d_ls: float,
        l_r_upper: float,
        m_r_upper: float,
        l_r_fore: float,
        m_r_fore: float,
        l_l_upper: float,
        m_l_upper: float,
        l_l_fore: float,
        m_l_fore: float,
        l_club: float,
        m_club: float,
        m_clubhead: float,
        grip_right: float,
        grip_left: float,
        g: float = 9.81,
    ):
        self.params = GolferParams(
            l_hub=l_hub,
            m_hub=m_hub,
            d_rs=d_rs,
            d_ls=d_ls,
            l_r_upper=l_r_upper,
            m_r_upper=m_r_upper,
            l_r_fore=l_r_fore,
            m_r_fore=m_r_fore,
            l_l_upper=l_l_upper,
            m_l_upper=m_l_upper,
            l_l_fore=l_l_fore,
            m_l_fore=m_l_fore,
            l_club=l_club,
            m_club=m_club,
            m_clubhead=m_clubhead,
            grip_right=grip_right,
            grip_left=grip_left,
            g=g,
        )
        self.use_native = HAS_NATIVE

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Compute the 8x8 mass matrix M(q)."""
        if self.use_native:
            try:
                result = pendulum_core.py_golfer_mass_matrix(q.tolist(), self.params.to_rust())
                return np.array(result, dtype=np.float64)
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    "Rust golfer mass_matrix call failed (%s), falling back to NumPy: %s",
                    type(e).__name__,
                    e,
                )

        # NumPy fallback would be implemented by porting the Rust analytical code
        raise NotImplementedError("NumPy fallback for golfer mass matrix not yet implemented")

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Compute the gravity vector G(q)."""
        if self.use_native:
            try:
                result = pendulum_core.py_golfer_gravity_vector(
                    q.tolist(), self.params.to_rust()
                )
                return np.array(result, dtype=np.float64)
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning("Rust golfer gravity_vector call failed (%s)", type(e).__name__)

        raise NotImplementedError("NumPy fallback for golfer gravity not yet implemented")

    def forward_kinematics(self, q: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Compute forward kinematics."""
        if self.use_native:
            try:
                result = pendulum_core.py_golfer_forward_kinematics(
                    q.tolist(), self.params.to_rust()
                )
                return {k: tuple(v) for k, v in result.items()}
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    "Rust golfer forward_kinematics call failed (%s)", type(e).__name__
                )

        raise NotImplementedError("NumPy fallback for golfer FK not yet implemented")

    def constraint_vector(self, q: np.ndarray) -> np.ndarray:
        """Compute the constraint vector Φ(q)."""
        if self.use_native:
            try:
                result = pendulum_core.py_golfer_constraint_vector(
                    q.tolist(), self.params.to_rust()
                )
                return np.array(result, dtype=np.float64)
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    "Rust golfer constraint_vector call failed (%s)", type(e).__name__
                )

        raise NotImplementedError("NumPy fallback for constraints not yet implemented")

    def constraint_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute the constraint Jacobian ∂Φ/∂q."""
        if self.use_native:
            try:
                result = pendulum_core.py_golfer_constraint_jacobian(
                    q.tolist(), self.params.to_rust()
                )
                return np.array(result, dtype=np.float64)
            except (RuntimeError, AttributeError, TypeError) as e:
                logger.warning(
                    "Rust golfer constraint_jacobian call failed (%s)", type(e).__name__
                )

        raise NotImplementedError("NumPy fallback for constraint Jacobian not yet implemented")


def get_native_info() -> Dict[str, object]:
    """Get information about native library availability."""
    return {
        "has_native": HAS_NATIVE,
        "error": NATIVE_ERROR,
        "module_version": (
            getattr(pendulum_core, "__version__", "unknown") if HAS_NATIVE else None
        ),
    }


__all__ = [
    "DoublePendulum",
    "DoublePendulumParams",
    "Golfer",
    "GolferParams",
    "HAS_NATIVE",
    "get_native_info",
]
