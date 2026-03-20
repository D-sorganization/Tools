# mypy: ignore-errors
"""SO(3) and SE(3) rotation/transformation helpers (Modern Robotics — Lynch & Park).

Internal submodule extracted from modern_robotics.py to keep file size within
the 1200-line budget.  Import these symbols via ``rotation_converter.modern_robotics``
(the public shim) rather than directly from this private module.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from rotation_converter._contracts import ensure, require, require_finite

# ---------------------------------------------------------------------------
# Internal helpers (DRY — shared across multiple functions)
# ---------------------------------------------------------------------------


def _near_zero(val: float, tol: float = 1e-12) -> bool:
    """Check if a scalar is effectively zero."""
    return abs(val) < tol


# ===========================================================================
# SO(3) — 3D Rotation helpers
# ===========================================================================


def VecToso3(omega: Any) -> np.ndarray:
    """Convert a 3-vector to a 3x3 skew-symmetric matrix [omega].

    Args:
        omega: 3-vector angular velocity.

    Returns:
        3x3 skew-symmetric matrix in so(3).
    """
    omega = np.asarray(omega, dtype=float)
    require(omega.shape == (3,), "omega must have 3 elements", omega.shape)
    return np.array(
        [
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0],
        ]
    )


def so3ToVec(so3mat: Any) -> np.ndarray:
    """Extract the 3-vector from a 3x3 skew-symmetric matrix.

    Args:
        so3mat: 3x3 skew-symmetric matrix.

    Returns:
        3-vector omega.
    """
    so3mat = np.asarray(so3mat, dtype=float)
    require(so3mat.shape == (3, 3), "so3 matrix must be 3x3", so3mat.shape)
    return np.array([so3mat[2, 1], so3mat[0, 2], so3mat[1, 0]])


def MatrixExp3(so3mat: Any) -> np.ndarray:
    """Compute the matrix exponential of an so(3) matrix -> SO(3).

    Implements Rodrigues' formula: if so3mat = [omega_hat]*theta,
    R = I + sin(theta)*[omega_hat] + (1-cos(theta))*[omega_hat]^2

    Args:
        so3mat: 3x3 so(3) matrix (skew-symmetric * angle).

    Returns:
        3x3 rotation matrix in SO(3).
    """
    so3mat = np.asarray(so3mat, dtype=float)
    require(so3mat.shape == (3, 3), "so(3) matrix must be 3x3")
    require_finite(so3mat, "so(3) matrix")

    omega_vec = so3ToVec(so3mat)
    theta = float(np.linalg.norm(omega_vec))

    if _near_zero(theta):
        return np.eye(3)

    omega_hat = so3mat / theta
    R = (
        np.eye(3)
        + math.sin(theta) * omega_hat
        + (1.0 - math.cos(theta)) * (omega_hat @ omega_hat)
    )

    ensure(abs(np.linalg.det(R) - 1.0) < 1e-9, "result must be SO(3)")
    return R  # type: ignore[no-any-return]


def MatrixLog3(R: Any) -> np.ndarray:
    """Compute the matrix logarithm of SO(3) -> so(3).

    Args:
        R: 3x3 rotation matrix in SO(3).

    Returns:
        3x3 skew-symmetric matrix in so(3) such that exp(result) = R.
    """
    R = np.asarray(R, dtype=float)
    require(R.shape == (3, 3), "rotation matrix must be 3x3")
    require_finite(R, "rotation matrix")

    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)

    if _near_zero(cos_theta - 1.0):
        # theta ≈ 0 -> near identity
        return np.zeros((3, 3))

    if _near_zero(cos_theta + 1.0):
        # theta ≈ pi
        theta = math.pi
        # Find the column of R + I with largest norm
        RpI = R + np.eye(3)
        col_norms = [np.linalg.norm(RpI[:, i]) for i in range(3)]
        best_col = int(np.argmax(col_norms))
        omega = RpI[:, best_col] / np.linalg.norm(RpI[:, best_col])
        return VecToso3(omega * theta)

    theta = math.acos(cos_theta)
    omega_hat = (R - R.T) / (2.0 * math.sin(theta))
    return omega_hat * theta  # type: ignore[no-any-return]


# ===========================================================================
# SE(3) — Rigid body transformation helpers
# ===========================================================================


def VecTose3(V: Any) -> np.ndarray:
    """Convert a 6-vector spatial velocity to a 4x4 se(3) matrix.

    V = [omega; v] -> [[omega] v; 0 0]

    Args:
        V: 6-vector [omega_1, omega_2, omega_3, v_1, v_2, v_3].

    Returns:
        4x4 matrix in se(3).
    """
    V = np.asarray(V, dtype=float)
    require(V.shape == (6,), "spatial velocity must have 6 elements", V.shape)
    M = np.zeros((4, 4))
    M[:3, :3] = VecToso3(V[:3])
    M[:3, 3] = V[3:]
    return M


def se3ToVec(se3mat: Any) -> np.ndarray:
    """Extract the 6-vector from a 4x4 se(3) matrix.

    Args:
        se3mat: 4x4 matrix in se(3).

    Returns:
        6-vector [omega; v].
    """
    se3mat = np.asarray(se3mat, dtype=float)
    require(se3mat.shape == (4, 4), "se(3) matrix must be 4x4")
    return np.concatenate([so3ToVec(se3mat[:3, :3]), se3mat[:3, 3]])


def TransToRp(T: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract rotation matrix R and position vector p from SE(3) matrix T.

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        Tuple of (R, p) where R is 3x3 and p is 3-vector.
    """
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "transform must be 4x4")
    return T[:3, :3].copy(), T[:3, 3].copy()


def RpToTrans(R: Any, p: Any) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from rotation matrix R and position p.

    Args:
        R: 3x3 rotation matrix.
        p: 3-vector position.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    R = np.asarray(R, dtype=float)
    p = np.asarray(p, dtype=float)
    require(R.shape == (3, 3), "R must be 3x3")
    require(p.shape == (3,), "p must have 3 elements")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def TransInv(T: Any) -> np.ndarray:
    """Compute the inverse of an SE(3) transformation matrix.

    Uses the efficient formula: T^-1 = [R^T  -R^T*p; 0 1].

    Args:
        T: 4x4 SE(3) matrix.

    Returns:
        4x4 inverse transformation.
    """
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "transform must be 4x4")
    R, p = TransToRp(T)
    Rt = R.T
    T_inv = np.eye(4)
    T_inv[:3, :3] = Rt
    T_inv[:3, 3] = -Rt @ p
    return T_inv


def _Adjoint(T: Any) -> np.ndarray:
    """6x6 adjoint representation of SE(3) matrix T.

    Ad_T = [R  0; [p]R  R]

    Used internally for Jacobian computation.
    """
    T = np.asarray(T, dtype=float)
    R, p = TransToRp(T)
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, 3:] = R
    Ad[3:, :3] = VecToso3(p) @ R
    return Ad


def MatrixExp6(se3mat: Any) -> np.ndarray:
    """Compute the matrix exponential of an se(3) matrix -> SE(3).

    If se3mat = [S]*theta where S = [omega; v] with ||omega||=1:
        T = [exp([omega]*theta)   G(theta)*v; 0  1]
    If omega = 0 (pure translation):
        T = [I  v*theta; 0  1]

    Args:
        se3mat: 4x4 matrix in se(3).

    Returns:
        4x4 matrix in SE(3).
    """
    se3mat = np.asarray(se3mat, dtype=float)
    require(se3mat.shape == (4, 4), "se(3) matrix must be 4x4")
    require_finite(se3mat, "se(3) matrix")

    omega_mat = se3mat[:3, :3]
    omega_vec = so3ToVec(omega_mat)
    v = se3mat[:3, 3]
    theta = float(np.linalg.norm(omega_vec))

    T = np.eye(4)

    if _near_zero(theta):
        # Pure translation
        T[:3, 3] = v
        return T

    # omega_mat = [omega_hat] * theta, v_full = v * theta
    omega_hat = omega_mat / theta  # unit skew-symmetric
    v_unit = v / theta  # v component of the unit twist

    R = MatrixExp3(omega_mat)
    # G(theta) from Lynch & Park Eq. 3.84
    G = (
        np.eye(3) * theta
        + (1.0 - math.cos(theta)) * omega_hat
        + (theta - math.sin(theta)) * (omega_hat @ omega_hat)
    )
    T[:3, :3] = R
    T[:3, 3] = G @ v_unit

    ensure(abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-9, "result must be SE(3)")
    return T


def MatrixLog6(T: Any) -> np.ndarray:
    """Compute the matrix logarithm of SE(3) -> se(3).

    Args:
        T: 4x4 SE(3) matrix.

    Returns:
        4x4 matrix in se(3) such that exp(result) = T.
    """
    T = np.asarray(T, dtype=float)
    require(T.shape == (4, 4), "SE(3) matrix must be 4x4")
    require_finite(T, "SE(3) matrix")

    R, p = TransToRp(T)
    omega_mat = MatrixLog3(R)
    omega_vec = so3ToVec(omega_mat)
    theta = float(np.linalg.norm(omega_vec))

    result = np.zeros((4, 4))

    if _near_zero(theta):
        # Pure translation
        result[:3, 3] = p
        return result

    omega_hat = omega_mat / theta
    G_inv = (
        np.eye(3) / theta
        - omega_hat / 2.0
        + (1.0 / theta - 1.0 / (2.0 * math.tan(theta / 2.0))) * (omega_hat @ omega_hat)
    )

    result[:3, :3] = omega_mat
    # G_inv @ p = v_unit; the se(3) matrix stores v_unit * theta
    result[:3, 3] = (G_inv @ p) * theta
    return result


# ===========================================================================
# Legacy SO(3)/SE(3) utility functions (auto-imported from original textbook code)
# ===========================================================================


def Normalize(V):
    """Normalizes a vector

    :param V: A vector
    :return: A unit vector pointing in the same direction as z

    Example Input:
        V = np.array([1, 2, 3])
    Output:
        np.array([0.26726124, 0.53452248, 0.80178373])
    """
    return V / np.linalg.norm(V)


def RotInv(R):
    """Inverts a rotation matrix

    :param R: A rotation matrix
    :return: The inverse of R

    Example Input:
        R = np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
    Output:
        np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
    """
    return np.array(R).T


def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form

    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle

    Example Input:
        expc3 = np.array([1, 2, 3])
    Output:
        (np.array([0.26726124, 0.53452248, 0.80178373]), 3.7416573867739413)
    """
    return (Normalize(expc3), np.linalg.norm(expc3))


def Adjoint(T):
    """Computes the adjoint representation of a homogeneous transformation
    matrix

    :param T: A homogeneous transformation matrix
    :return: The 6x6 adjoint representation [AdT] of T

    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1, 0,  0, 0, 0,  0],
                  [0, 0, -1, 0, 0,  0],
                  [0, 1,  0, 0, 0,  0],
                  [0, 0,  3, 1, 0,  0],
                  [3, 0,  0, 0, 0, -1],
                  [0, 0,  0, 0, 1,  0]])
    """
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))], np.c_[np.dot(VecToso3(p), R), R]]


def ScrewToAxis(q, s, h):
    """Takes a parametric description of a screw axis and converts it to a
    normalized screw axis

    :param q: A point lying on the screw axis
    :param s: A unit vector in the direction of the screw axis
    :param h: The pitch of the screw axis
    :return: A normalized screw axis described by the inputs

    Example Input:
        q = np.array([3, 0, 0])
        s = np.array([0, 0, 1])
        h = 2
    Output:
        np.array([0, 0, 1, 0, -3, 2])
    """
    return np.r_[s, np.cross(q, s) + np.dot(h, s)]


def AxisAng6(expc6):
    """Converts a 6-vector of exponential coordinates into screw axis-angle
    form

    :param expc6: A 6-vector of exponential coordinates for rigid-body motion
                  S*theta
    :return S: The corresponding normalized screw axis
    :return theta: The distance traveled along/about S

    Example Input:
        expc6 = np.array([1, 0, 0, 1, 2, 3])
    Output:
        (np.array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), 1.0)
    """
    theta = np.linalg.norm([expc6[0], expc6[1], expc6[2]])
    if _near_zero(theta):
        theta = np.linalg.norm([expc6[3], expc6[4], expc6[5]])
    return (np.array(expc6 / theta), theta)


def ProjectToSO3(mat):
    """Returns a projection of mat into SO(3)

    :param mat: A matrix near SO(3) to project to SO(3)
    :return: The closest matrix to R that is in SO(3)
    Projects a matrix mat to the closest matrix in SO(3) using singular-value
    decomposition (see
    http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
    This function is only appropriate for matrices close to SO(3).

    Example Input:
        mat = np.array([[ 0.675,  0.150,  0.720],
                        [ 0.370,  0.771, -0.511],
                        [-0.630,  0.619,  0.472]])
    Output:
        np.array([[ 0.67901136,  0.14894516,  0.71885945],
                  [ 0.37320708,  0.77319584, -0.51272279],
                  [-0.63218672,  0.61642804,  0.46942137]])
    """
    U, s, Vh = np.linalg.svd(mat)
    R = np.dot(U, Vh)
    if np.linalg.det(R) < 0:
        # In this case the result may be far from mat.
        R[:, 2] = -R[:, 2]
    return R


def ProjectToSE3(mat):
    """Returns a projection of mat into SE(3)

    :param mat: A 4x4 matrix to project to SE(3)
    :return: The closest matrix to T that is in SE(3)
    Projects a matrix mat to the closest matrix in SE(3) using singular-value
    decomposition (see
    http://hades.mech.northwestern.edu/index.php/Modern_Robotics_Linear_Algebra_Review).
    This function is only appropriate for matrices close to SE(3).

    Example Input:
        mat = np.array([[ 0.675,  0.150,  0.720,  1.2],
                        [ 0.370,  0.771, -0.511,  5.4],
                        [-0.630,  0.619,  0.472,  3.6],
                        [ 0.003,  0.002,  0.010,  0.9]])
    Output:
        np.array([[ 0.67901136,  0.14894516,  0.71885945,  1.2 ],
                  [ 0.37320708,  0.77319584, -0.51272279,  5.4 ],
                  [-0.63218672,  0.61642804,  0.46942137,  3.6 ],
                  [ 0.        ,  0.        ,  0.        ,  1.  ]])
    """
    mat = np.array(mat)
    return RpToTrans(ProjectToSO3(mat[:3, :3]), mat[:3, 3])


def DistanceToSO3(mat):
    """Returns the Frobenius norm to describe the distance of mat from the
    SO(3) manifold

    :param mat: A 3x3 matrix
    :return: A quantity describing the distance of mat from the SO(3)
             manifold
    Computes the distance from mat to the SO(3) manifold using the following
    method:
    If det(mat) <= 0, return a large number.
    If det(mat) > 0, return norm(mat^T.mat - I).

    Example Input:
        mat = np.array([[ 1.0,  0.0,   0.0 ],
                        [ 0.0,  0.1,  -0.95],
                        [ 0.0,  1.0,   0.1 ]])
    Output:
        0.08835
    """
    if np.linalg.det(mat) > 0:
        return np.linalg.norm(np.dot(np.array(mat).T, mat) - np.eye(3))
    else:
        return 1e9


def DistanceToSE3(mat):
    """Returns the Frobenius norm to describe the distance of mat from the
    SE(3) manifold

    :param mat: A 4x4 matrix
    :return: A quantity describing the distance of mat from the SE(3)
              manifold
    Computes the distance from mat to the SE(3) manifold using the following
    method:
    Compute the determinant of matR, the top 3x3 submatrix of mat.
    If det(matR) <= 0, return a large number.
    If det(matR) > 0, replace the top 3x3 submatrix of mat with matR^T.matR,
    and set the first three entries of the fourth column of mat to zero. Then
    return norm(mat - I).

    Example Input:
        mat = np.array([[ 1.0,  0.0,   0.0,   1.2 ],
                        [ 0.0,  0.1,  -0.95,  1.5 ],
                        [ 0.0,  1.0,   0.1,  -0.9 ],
                        [ 0.0,  0.0,   0.1,   0.98 ]])
    Output:
        0.134931
    """
    matR = np.array(mat)[0:3, 0:3]
    if np.linalg.det(matR) > 0:
        return np.linalg.norm(
            np.r_[
                np.c_[np.dot(np.transpose(matR), matR), np.zeros((3, 1))],
                [np.array(mat)[3, :]],
            ]
            - np.eye(4)
        )
    else:
        return 1e9


def TestIfSO3(mat):
    """Returns true if mat is close to or on the manifold SO(3)

    :param mat: A 3x3 matrix
    :return: True if mat is very close to or in SO(3), false otherwise
    Computes the distance d from mat to the SO(3) manifold using the
    following method:
    If det(mat) <= 0, d = a large number.
    If det(mat) > 0, d = norm(mat^T.mat - I).
    If d is close to zero, return true. Otherwise, return false.

    Example Input:
        mat = np.array([[1.0, 0.0,  0.0 ],
                        [0.0, 0.1, -0.95],
                        [0.0, 1.0,  0.1 ]])
    Output:
        False
    """
    return abs(DistanceToSO3(mat)) < 1e-3


def TestIfSE3(mat):
    """Returns true if mat is close to or on the manifold SE(3)

    :param mat: A 4x4 matrix
    :return: True if mat is very close to or in SE(3), false otherwise
    Computes the distance d from mat to the SE(3) manifold using the
    following method:
    Compute the determinant of the top 3x3 submatrix of mat.
    If det(mat) <= 0, d = a large number.
    If det(mat) > 0, replace the top 3x3 submatrix of mat with mat^T.mat, and
    set the first three entries of the fourth column of mat to zero.
    Then d = norm(T - I).
    If d is close to zero, return true. Otherwise, return false.

    Example Input:
        mat = np.array([[1.0, 0.0,   0.0,  1.2],
                        [0.0, 0.1, -0.95,  1.5],
                        [0.0, 1.0,   0.1, -0.9],
                        [0.0, 0.0,   0.1, 0.98]])
    Output:
        False
    """
    return abs(DistanceToSE3(mat)) < 1e-3
