"""Twist and screw axis conversion module.

Representations:
- Twist vector: 6-vector [omega; v] where omega is angular, v is linear velocity
- se(3) matrix: 4x4 matrix form of twist (Lie algebra of SE(3))
- Screw axis: geometric parameterisation (axis direction, point, pitch)
- SE(3) homogeneous transformation matrix: 4x4 rigid body transform

Uses the matrix exponential / logarithm for SE(3) <-> twist+angle.
Follows Lynch & Park "Modern Robotics" conventions.

DbC: preconditions validate inputs, postconditions verify SE(3) membership.
DRY: reuses skew-symmetric and Rodrigues from core module.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rotation_converter._contracts import (
    ensure,
    require,
    require_finite,
    require_unit_vector,
)
from rotation_converter.core import (
    _skew_symmetric,
    _validate_rotation_matrix,
    rotation_matrix_to_axis_angle,
)

# ---------------------------------------------------------------------------
# Twist vector <-> se(3) matrix
# ---------------------------------------------------------------------------


def twist_vector_to_se3_matrix(xi: Any) -> np.ndarray:
    """Convert a 6-vector twist [omega; v] to a 4x4 se(3) matrix.

    The se(3) matrix has the form::

        [  [omega]x   v ]
        [    0        0 ]

    Precondition: xi has 6 elements.
    Postcondition: result is 4x4 with zero bottom row.
    """
    xi = np.asarray(xi, dtype=float)
    require(xi.shape == (6,), "twist must have 6 elements", xi.shape)
    require_finite(xi, "twist")

    omega = xi[:3]
    v = xi[3:]
    M = np.zeros((4, 4))
    M[:3, :3] = _skew_symmetric(omega)
    M[:3, 3] = v

    ensure(
        np.allclose(M[3, :], 0),
        "bottom row of se(3) matrix must be zero",
    )
    return M


def se3_matrix_to_twist_vector(M: Any) -> np.ndarray:
    """Convert a 4x4 se(3) matrix to a 6-vector twist [omega; v].

    Precondition: M is 4x4 with zero bottom row.
    Postcondition: result has 6 elements.
    """
    M = np.asarray(M, dtype=float)
    require(M.shape == (4, 4), "se(3) matrix must be 4x4", M.shape)
    require(
        np.allclose(M[3, :], 0),
        "bottom row of se(3) matrix must be zero",
    )

    omega = np.array([M[2, 1], M[0, 2], M[1, 0]])
    v = M[:3, 3]
    result = np.concatenate([omega, v])

    ensure(result.shape == (6,), "result must have 6 elements")
    return result


# ---------------------------------------------------------------------------
# Twist + angle <-> SE(3) homogeneous matrix (matrix exponential)
# ---------------------------------------------------------------------------


def twist_angle_to_homogeneous(xi: Any, theta: float) -> np.ndarray:
    """Compute the matrix exponential exp([xi]*theta) -> SE(3).

    For a twist xi = [omega; v] with ||omega|| = 1 (rotation case)::

        T = [ e^([omega]x * theta)    G(theta) * v ]
            [       0                       1       ]

    where G(theta) = I*theta + (1-cos(theta))[omega]x + (theta-sin(theta))[omega]x^2

    For pure translation (omega = 0, ||v|| = 1)::

        T = [ I    v*theta ]
            [ 0       1    ]

    Precondition: xi has 6 elements, for rotation case ||omega||=1.
    Postcondition: T is in SE(3).
    """
    xi = np.asarray(xi, dtype=float)
    require(xi.shape == (6,), "twist must have 6 elements", xi.shape)
    require_finite(xi, "twist")

    omega = xi[:3]
    v = xi[3:]
    omega_norm = np.linalg.norm(omega)

    T = np.eye(4)

    if omega_norm < 1e-12:
        # Pure translation
        T[:3, 3] = v * theta
    else:
        # Rotation (possibly with translation)
        require(
            bool(abs(omega_norm - 1.0) < 1e-6),
            "angular twist component must be unit length for rotation",
            omega_norm,
        )
        K = _skew_symmetric(omega)
        R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
        G = (
            np.eye(3) * theta
            + (1.0 - math.cos(theta)) * K
            + (theta - math.sin(theta)) * (K @ K)
        )
        T[:3, :3] = R
        T[:3, 3] = G @ v

    # Postcondition: SE(3)
    R_out = T[:3, :3]
    ensure(
        abs(np.linalg.det(R_out) - 1.0) < 1e-9,
        "rotation part must have det=+1",
    )
    ensure(np.allclose(T[3, :], [0, 0, 0, 1]), "bottom row must be [0,0,0,1]")
    return T


def homogeneous_to_twist_angle(T: Any) -> tuple[np.ndarray, float]:
    """Compute the matrix logarithm of an SE(3) matrix -> (twist, angle).

    Decomposes T into twist xi and angle theta such that
    exp([xi]*theta) = T.

    Precondition: T is in SE(3).
    Postcondition: twist has 6 elements, theta >= 0.
    """
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "homogeneous matrix must be 4x4", T.shape)
    require(
        np.allclose(T[3, :], [0, 0, 0, 1]),
        "bottom row must be [0,0,0,1]",
    )

    R = T[:3, :3]
    p = T[:3, 3]

    # Check if R is identity (pure translation)
    if np.allclose(R, np.eye(3), atol=1e-9):
        p_norm = np.linalg.norm(p)
        if p_norm < 1e-12:
            # Identity transform
            xi, theta = np.zeros(6), 0.0
        else:
            # Pure translation
            v_hat = p / p_norm
            xi, theta = np.concatenate([np.zeros(3), v_hat]), float(p_norm)
    else:
        # General case: extract axis-angle from R
        _validate_rotation_matrix(R)
        axis, theta = rotation_matrix_to_axis_angle(R)

        if theta < 1e-12:
            # Near-identity rotation, treat as pure translation
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-12:
                xi, theta = np.zeros(6), 0.0
            else:
                v_hat = p / p_norm
                xi, theta = np.concatenate([np.zeros(3), v_hat]), float(p_norm)
        else:
            omega = axis
            K = _skew_symmetric(omega)
            # G_inv = (1/theta)*I - 0.5*K + (1/theta - 0.5*cot(theta/2))*K^2
            cot_half = math.cos(theta / 2) / math.sin(theta / 2)
            G_inv = (
                (1.0 / theta) * np.eye(3)
                - 0.5 * K
                + (1.0 / theta - 0.5 * cot_half) * (K @ K)
            )
            v = G_inv @ p
            xi = np.concatenate([omega, v])

    ensure(xi.shape == (6,), "result twist must have 6 elements")
    ensure(theta >= 0, "angle must be non-negative")
    return xi, float(theta)


# ---------------------------------------------------------------------------
# Twist <-> Screw axis
# ---------------------------------------------------------------------------


def twist_to_screw(xi: Any) -> dict[str, Any]:
    """Decompose a twist into screw axis parameters.

    Returns a dict with keys:
    - ``axis``: unit direction of screw axis (3-vector)
    - ``point``: a point on the screw axis (3-vector), zero for pure translation
    - ``pitch``: translation per radian (float, inf for pure translation)

    For rotation case (||omega|| > 0):
        pitch = omega . v / ||omega||^2
        axis = omega / ||omega||
        point q satisfies: v = -omega x q + pitch * omega

    For pure translation (omega = 0):
        axis = v / ||v||
        pitch = inf
        point = [0, 0, 0]
    """
    xi = np.asarray(xi, dtype=float)
    require(xi.shape == (6,), "twist must have 6 elements", xi.shape)
    require_finite(xi, "twist")

    omega = xi[:3]
    v = xi[3:]
    omega_norm = np.linalg.norm(omega)

    if omega_norm < 1e-12:
        # Pure translation
        v_norm = np.linalg.norm(v)
        require(bool(v_norm > 1e-12), "twist cannot be zero for screw decomposition")
        return {
            "axis": v / v_norm,
            "point": np.zeros(3),
            "pitch": float("inf"),
        }

    axis = omega / omega_norm
    pitch = float(np.dot(omega, v) / (omega_norm**2))

    # Find point on axis: q = omega x v / ||omega||^2
    q = np.cross(omega, v) / (omega_norm**2)

    return {
        "axis": axis,
        "point": q,
        "pitch": pitch,
    }


def screw_to_twist(screw: dict[str, Any]) -> np.ndarray:
    """Convert screw axis parameters to a twist vector [omega; v].

    Args:
        screw: dict with ``axis``, ``point``, ``pitch`` keys.

    For finite pitch:
        omega = axis (unit)
        v = -omega x point + pitch * omega

    For infinite pitch (pure translation):
        omega = [0, 0, 0]
        v = axis (unit)
    """
    axis = np.asarray(screw["axis"], dtype=float)
    point = np.asarray(screw["point"], dtype=float)
    pitch = screw["pitch"]

    require(axis.shape == (3,), "screw axis must have 3 elements")

    if pitch == float("inf"):
        # Pure translation — normalize axis to unit direction
        axis_norm = np.linalg.norm(axis)
        require(
            bool(axis_norm > 1e-12), "screw axis must be non-zero for pure translation"
        )
        omega = np.zeros(3)
        v = axis / axis_norm
    else:
        # Rotation/helical — axis must be unit for valid twist
        require_unit_vector(axis, "screw axis")
        omega = axis
        v = np.cross(-omega, point) + pitch * omega

    xi = np.concatenate([omega, v])
    ensure(xi.shape == (6,), "result must have 6 elements")
    return xi


# ---------------------------------------------------------------------------
# Adjoint representation
# ---------------------------------------------------------------------------


def adjoint_representation(T: Any) -> np.ndarray:
    """Compute the 6x6 adjoint representation of an SE(3) matrix.

    The adjoint maps twists between frames::

        Ad_T = [ R    [p]x R ]
               [ 0      R    ]

    Precondition: T is in SE(3).
    Postcondition: result is 6x6.
    """
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "SE(3) matrix must be 4x4")
    require(
        np.allclose(T[3, :], [0, 0, 0, 1]),
        "bottom row must be [0,0,0,1]",
    )

    R = T[:3, :3]
    p = T[:3, 3]

    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = _skew_symmetric(p) @ R

    ensure(Ad.shape == (6, 6), "adjoint must be 6x6")
    return Ad
