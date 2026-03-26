# mypy: disable-error-code="no-any-return"
"""Advanced Kinematics.

Provides representations and operations for Denavit-Hartenberg parameters
and Dual Quaternions for comprehensive robotic kinematics tracking.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require, require_finite
from rotation_converter.core import normalize_quaternion, quaternion_multiply


def dh_to_matrix(
    theta: float, d: float, a: float, alpha: float, modified: bool = False
) -> np.ndarray:
    """Convert Denavit-Hartenberg parameters to a 4x4 homogenous matrix.

    Args:
        theta: Joint rotation about z-axis (rad)
        d: Link offset along z-axis
        a: Link length along x-axis
        alpha: Link twist about x-axis (rad)
        modified: If True, uses modified Craig DH parameters. If False,
                  uses standard Spong DH parameters.

    Returns:
        4x4 homogeneous transformation matrix SE(3).
    """
    if not (theta is not None):
        raise ValueError("theta must be provided")
    require_finite(theta, "dh_theta")
    require_finite(d, "dh_d")
    require_finite(a, "dh_a")
    require_finite(alpha, "dh_alpha")

    ct = math.cos(theta)
    st = math.sin(theta)
    ca = math.cos(alpha)
    sa = math.sin(alpha)

    # Clean up near-zero floats for perfect orthogonality
    ct = 0.0 if abs(ct) < 1e-12 else ct
    st = 0.0 if abs(st) < 1e-12 else st
    ca = 0.0 if abs(ca) < 1e-12 else ca
    sa = 0.0 if abs(sa) < 1e-12 else sa

    if modified:
        # Modified DH (Craig): i-1 to i
        # Rot(X, alpha_{i-1}) * Trans(X, a_{i-1}) * Rot(Z, theta_i) * Trans(Z, d_i)
        T = np.array(
            [
                [ct, -st, 0, a],
                [st * ca, ct * ca, -sa, -d * sa],
                [st * sa, ct * sa, ca, d * ca],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    else:
        # Standard DH (Spong): i-1 to i
        # Rot(Z, theta_i) * Trans(Z, d_i) * Trans(X, a_i) * Rot(X, alpha_i)
        T = np.array(
            [
                [ct, -st * ca, st * sa, a * ct],
                [st, ct * ca, -ct * sa, a * st],
                [0.0, sa, ca, d],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    ensure(T.shape == (4, 4), "Output must be 4x4 matrix")
    return T


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical Linear Interpolation (slerp) between two unit quaternions.

    Args:
        q1: Unit quaternion (w, x, y, z) at t=0.
        q2: Unit quaternion (w, x, y, z) at t=1.
        t: Interpolation parameter in range [0, 1].

    Returns:
        Interpolated unit quaternion (w, x, y, z).
    """
    if not (q1 is not None):
        raise ValueError("q1 must be provided")
    require(0.0 <= t <= 1.0, f"Interpolation parameter t={t} must be in [0, 1]")
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)

    dot = np.dot(q1, q2)

    # If the dot product is negative, slerp takes the long way around.
    # We negate q2 to take the shortest path instead.
    if dot < 0.0:
        q2 = -q2
        dot = -dot

    # If the inputs are too close, linearly interpolate and normalize
    # to avoid division by zero
    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return normalize_quaternion(result)

    # Standard slerp
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return normalize_quaternion((s1 * q1) + (s2 * q2))


class DualQuaternion:
    """A Dual Quaternion representation of rigid body displacement.

    Combines rotation (real part) and translation (dual part) into a single
    mathematical entity structured as: Q = q_r + eps * q_d
    """

    def __init__(self, qr: Any, qd: Any) -> None:
        """Initialize dual quaternion from real and dual quaternions.

        Args:
            qr: Real quaternion (w, x, y, z) representing rotation.
            qd: Dual quaternion (w, x, y, z) representing translation scaled.
        """
        self._qr = np.asarray(qr, dtype=float).flatten()
        self._qd = np.asarray(qd, dtype=float).flatten()
        require(self._qr.shape == (4,), "Real quaternion must be a 4-vector.")
        require(self._qd.shape == (4,), "Dual quaternion must be a 4-vector.")

    @classmethod
    def from_translation_rotation(
        cls, translation: Any, rotation_quaternion: Any
    ) -> DualQuaternion:
        """Construct from a translation vector and rotation quaternion."""
        t = np.asarray(translation, dtype=float).flatten()
        require(t.shape == (3,), "Translation must be a 3-vector.")
        qr = normalize_quaternion(rotation_quaternion)

        # Dual part = (1/2) * t * qr
        # Represent t as a pure quaternion for multiplication: (0, tx, ty, tz)
        t_quat = np.array([0.0, t[0], t[1], t[2]])
        qd = 0.5 * quaternion_multiply(t_quat, qr)

        return cls(qr, qd)

    @property
    def real(self) -> np.ndarray:
        """Return the real quaternion component."""
        return self._qr.copy()

    @property
    def dual(self) -> np.ndarray:
        """Return the dual quaternion component."""
        return self._qd.copy()

    def multiply(self, other: DualQuaternion) -> DualQuaternion:
        """Multiply two dual quaternions.

        Calculates: (qr1 * qr2) + eps * (qr1 * qd2 + qd1 * qr2)
        """
        if not (other is not None):
            raise ValueError("other must be provided")
        qr_new = quaternion_multiply(self._qr, other._qr)
        qd_new = quaternion_multiply(self._qr, other._qd) + quaternion_multiply(self._qd, other._qr)
        return DualQuaternion(qr_new, qd_new)

    def extract_translation(self) -> np.ndarray:
        """Extract the translation 3-vector from the dual quaternion."""
        # t = 2 * qd * conjugate(qr)
        from rotation_converter.core import quaternion_conjugate

        qr_conj = quaternion_conjugate(self._qr)
        t_quat = 2.0 * quaternion_multiply(self._qd, qr_conj)
        return t_quat[1:]
