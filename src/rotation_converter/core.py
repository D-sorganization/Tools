"""Core rotation representation conversions.

Hub-and-spoke architecture with quaternion as the canonical hub.
All pairwise conversions route through quaternion when no direct
formula exists, keeping the implementation DRY.

Representations supported:
- Quaternion (w, x, y, z) — unit quaternion in Hamilton convention
- Rotation matrix — 3x3 SO(3)
- Euler angles — intrinsic rotations for 12 conventions
- Axis-angle — unit axis + scalar angle
- Rodrigues vector — axis * angle compact form

Design by Contract (DbC):
- Preconditions validate inputs (unit quaternion, SO(3) matrix, etc.)
- Postconditions verify outputs (unit norm, orthogonality, det=+1)
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

# ---------------------------------------------------------------------------
# Internal helpers (DRY — reused across multiple conversions)
# ---------------------------------------------------------------------------

# Euler axis index mapping (shared by euler_to_quaternion & quaternion_to_euler)
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix [v]x for cross-product."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )


def _elementary_quaternion(axis_char: str, angle: float) -> np.ndarray:
    """Quaternion for an elementary rotation about a single axis."""
    if not isinstance(axis_char, str):
        raise TypeError(f"axis_char must be a str, got {type(axis_char).__name__!r}")
    idx = _AXIS_INDEX[axis_char]
    half = angle / 2.0
    q = np.zeros(4)
    q[0] = math.cos(half)
    q[1 + idx] = math.sin(half)
    return q


def _validate_quaternion_array(q: Any, name: str = "quaternion") -> np.ndarray:
    """Convert to ndarray and validate shape/finiteness."""
    if not isinstance(name, str):
        raise TypeError(f"name must be a str, got {type(name).__name__!r}")
    q = np.asarray(q, dtype=float)
    require(q.shape == (4,), f"{name} must have 4 elements", q.shape)
    require_finite(q, name)
    return q  # type: ignore[no-any-return]


def _validate_unit_quaternion(q: np.ndarray, name: str = "quaternion") -> None:
    """Require that q is a unit quaternion."""
    if q is None:
        raise TypeError("q must be provided, got None")
    norm = np.linalg.norm(q)
    require(
        bool(abs(norm - 1.0) < 1e-6),
        f"{name} must be a unit quaternion (norm={norm:.6f})",
        norm,
    )


def _validate_rotation_matrix(R: Any, name: str = "rotation matrix") -> np.ndarray:
    """Convert, validate shape, orthogonality, and det=+1."""
    if not isinstance(name, str):
        raise TypeError(f"name must be a str, got {type(name).__name__!r}")
    R = np.asarray(R, dtype=float)
    require(R.shape == (3, 3), f"{name} must be 3x3", R.shape)
    require_finite(R, name)
    orth_err = np.max(np.abs(R @ R.T - np.eye(3)))
    require(
        bool(orth_err < 1e-6), f"{name} must be orthogonal (max err={orth_err:.2e})"
    )
    det = np.linalg.det(R)
    require(bool(abs(det - 1.0) < 1e-6), f"{name} must have det=+1 (got {det:.6f})")
    return R  # type: ignore[no-any-return]


# ===========================================================================
# Quaternion primitives
# ===========================================================================


def normalize_quaternion(q: Any) -> np.ndarray:
    """Normalize a quaternion to unit length.

    Precondition: q must not be the zero vector.
    Postcondition: result has unit norm.
    """
    q = _validate_quaternion_array(q)
    norm = np.linalg.norm(q)
    require(bool(norm > 1e-12), "cannot normalize zero quaternion", norm)
    result = q / norm
    ensure(bool(abs(np.linalg.norm(result) - 1.0) < 1e-12), "result must be unit norm")
    return result  # type: ignore[no-any-return]


def quaternion_conjugate(q: Any) -> np.ndarray:
    """Return the conjugate (w, -x, -y, -z) of a quaternion."""
    q = _validate_quaternion_array(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1: Any, q2: Any) -> np.ndarray:
    """Hamilton product of two quaternions.

    Postcondition: if both inputs are unit, output is unit.
    """
    q1 = _validate_quaternion_array(q1, "q1")
    q2 = _validate_quaternion_array(q2, "q2")
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# ===========================================================================
# Quaternion <-> Rotation Matrix
# ===========================================================================


def quaternion_to_rotation_matrix(q: Any) -> np.ndarray:
    """Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Precondition: q is a unit quaternion.
    Postcondition: R is in SO(3) (orthogonal, det=+1).
    """
    q = _validate_quaternion_array(q)
    _validate_unit_quaternion(q)

    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )

    ensure(
        bool(abs(np.linalg.det(R) - 1.0) < 1e-9),
        "result rotation matrix must have det=+1",
    )
    return R


def rotation_matrix_to_quaternion(R: Any) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.

    Precondition: R is in SO(3).
    Postcondition: result is a unit quaternion with w >= 0.
    """
    R = _validate_rotation_matrix(R)

    trace = R[0, 0] + R[1, 1] + R[2, 2]
    candidates = [trace, R[0, 0], R[1, 1], R[2, 2]]
    best = int(np.argmax(candidates))

    if best == 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif best == 1:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif best == 2:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    # Canonical form: w >= 0
    if q[0] < 0:  # type: ignore
        q = -q

    ensure(bool(abs(np.linalg.norm(q) - 1.0) < 1e-9), "result must be unit quaternion")
    return q  # type: ignore


# ===========================================================================
# Quaternion <-> Axis-Angle
# ===========================================================================


def axis_angle_to_quaternion(axis: Any, angle: float) -> np.ndarray:
    """Convert axis-angle to unit quaternion.

    Precondition: axis is a unit vector.
    Postcondition: result is a unit quaternion.
    """
    if not isinstance(angle, (int, float)):
        raise TypeError(f"angle must be a number, got {type(angle).__name__!r}")
    axis = np.asarray(axis, dtype=float)
    require(axis.shape == (3,), "axis must have 3 elements", axis.shape)
    require_finite(axis, "axis")
    require_unit_vector(axis, "axis")
    require_finite(np.array([angle]), "angle")

    half = angle / 2.0
    q = np.array(
        [
            math.cos(half),
            axis[0] * math.sin(half),
            axis[1] * math.sin(half),
            axis[2] * math.sin(half),
        ]
    )

    ensure(bool(abs(np.linalg.norm(q) - 1.0) < 1e-9), "result must be unit quaternion")
    return q


def quaternion_to_axis_angle(q: Any) -> tuple[np.ndarray, float]:
    """Convert a unit quaternion to axis-angle representation.

    Returns (axis, angle) where axis is a unit vector and angle in [0, pi].
    For identity rotation, returns ([1,0,0], 0.0).

    Precondition: q is a unit quaternion.
    Postcondition: angle in [0, pi], axis is unit.
    """
    q = _validate_quaternion_array(q)
    _validate_unit_quaternion(q)
    # Canonical form: w >= 0 so angle in [0, pi]
    if q[0] < 0:
        q = -q

    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    sin_half = math.sin(angle / 2.0)

    if abs(sin_half) < 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0
    else:
        axis = q[1:] / sin_half

    ensure(angle >= 0, "angle must be non-negative")
    return axis, float(angle)


# ===========================================================================
# Axis-Angle <-> Rotation Matrix (Rodrigues formula — direct, no quaternion)
# ===========================================================================


def axis_angle_to_rotation_matrix(axis: Any, angle: float) -> np.ndarray:
    """Convert axis-angle to rotation matrix via Rodrigues' rotation formula.

    R = I + sin(θ)[k]x + (1 - cos(θ))[k]x²

    Precondition: axis is a unit vector.
    Postcondition: R in SO(3).
    """
    if not isinstance(angle, (int, float)):
        raise TypeError(f"angle must be a number, got {type(angle).__name__!r}")
    axis = np.asarray(axis, dtype=float)
    require(axis.shape == (3,), "axis must have 3 elements")
    require_unit_vector(axis, "axis")

    K = _skew_symmetric(axis)
    R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)

    ensure(bool(abs(np.linalg.det(R) - 1.0) < 1e-9), "result must be SO(3)")
    return R  # type: ignore[no-any-return]


def rotation_matrix_to_axis_angle(R: Any) -> tuple[np.ndarray, float]:
    """Extract axis-angle from a rotation matrix.

    Precondition: R in SO(3).
    Postcondition: angle in [0, pi], axis is unit.
    """
    R = _validate_rotation_matrix(R)
    q = rotation_matrix_to_quaternion(R)
    return quaternion_to_axis_angle(q)


# ===========================================================================
# Quaternion <-> Rodrigues vector
# ===========================================================================


def quaternion_to_rodrigues(q: Any) -> np.ndarray:
    """Convert unit quaternion to Rodrigues vector (axis * angle).

    Postcondition: ||r|| = angle.
    """
    q = _validate_quaternion_array(q)
    _validate_unit_quaternion(q)
    axis, angle = quaternion_to_axis_angle(q)
    return axis * angle


def rodrigues_to_quaternion(r: Any) -> np.ndarray:
    """Convert Rodrigues vector to unit quaternion.

    Precondition: r has 3 elements.
    Postcondition: result is unit quaternion.
    """
    r = np.asarray(r, dtype=float)
    require(r.shape == (3,), "Rodrigues vector must have 3 elements", r.shape)
    require_finite(r, "rodrigues")

    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = r / angle
    return axis_angle_to_quaternion(axis, angle)


# ===========================================================================
# Quaternion <-> Euler angles (12 conventions)
# ===========================================================================

# All 12 Tait-Bryan + proper Euler conventions
_VALID_CONVENTIONS = {
    "xyz",
    "xzy",
    "yxz",
    "yzx",
    "zxy",
    "zyx",  # Tait-Bryan
    "xyx",
    "xzx",
    "yxy",
    "yzy",
    "zxz",
    "zyz",  # Proper Euler
}


def _validate_euler_convention(convention: str) -> None:
    require(
        convention.lower() in _VALID_CONVENTIONS,
        f"Unknown Euler convention '{convention}'. Valid: {sorted(_VALID_CONVENTIONS)}",
    )


def euler_to_quaternion(a: float, b: float, c: float, convention: str) -> np.ndarray:
    """Convert Euler angles to unit quaternion.

    Intrinsic rotations: R = R_first(a) * R_second(b) * R_third(c).

    Args:
        a, b, c: rotation angles in radians.
        convention: 3-char string like "xyz", "zyx", "zyz", etc.

    Precondition: convention is one of the 12 valid conventions.
    Postcondition: result is unit quaternion.
    """
    if not isinstance(a, (int, float)):
        raise TypeError(f"a must be a number, got {type(a).__name__!r}")
    _validate_euler_convention(convention)
    conv = convention.lower()
    q1 = _elementary_quaternion(conv[0], a)
    q2 = _elementary_quaternion(conv[1], b)
    q3 = _elementary_quaternion(conv[2], c)
    q = quaternion_multiply(quaternion_multiply(q1, q2), q3)
    return normalize_quaternion(q)


def quaternion_to_euler(q: Any, convention: str) -> tuple[float, float, float]:
    """Convert unit quaternion to Euler angles.

    Routes through rotation matrix for robust extraction.

    Precondition: q is unit quaternion, convention is valid.
    Postcondition: returned angles reproduce the same rotation.
    """
    if not isinstance(convention, str):
        raise TypeError(f"convention must be a str, got {type(convention).__name__!r}")
    q = _validate_quaternion_array(q)
    _validate_unit_quaternion(q)
    _validate_euler_convention(convention)

    R = quaternion_to_rotation_matrix(q)
    return _rotation_matrix_to_euler_impl(R, convention.lower())


def _rotation_matrix_to_euler_impl(
    R: np.ndarray, conv: str
) -> tuple[float, float, float]:
    """Extract Euler angles from R for a given convention.

    Handles both Tait-Bryan (e.g. xyz) and proper Euler (e.g. zyz).
    """
    if R is None:
        raise TypeError("R must be provided, got None")
    i, j, k = (_AXIS_INDEX[c] for c in conv)

    # Detect if this is a proper Euler convention (first == last axis)
    is_proper = conv[0] == conv[2]

    if is_proper:
        # Proper Euler angles: e.g. zyz, xyx
        # Middle axis is different from first and last
        # Use the formula for proper Euler decomposition
        # R = Ri(a) Rj(b) Ri(c)
        # Find the third axis index that is neither i nor j
        other = 3 - i - j  # works because 0+1+2=3
        # Sign factor: +1 if (i,j) is a cyclic pair (0->1, 1->2, 2->0), else -1
        sign = 1.0 if (j - i) % 3 == 1 else -1.0

        cb = np.clip(R[i, i], -1.0, 1.0)
        b = math.acos(cb)

        if abs(math.sin(b)) > 1e-8:
            # General case: extract a and c from the i-th column/row
            # R[j,i] = sign * sin(b) * sin(a), R[other,i] = -sign * sin(b) * cos(a)
            a = math.atan2(R[j, i], -sign * R[other, i])
            # R[i,j] = sign * sin(b) * sin(c), R[i,other] = sign * sin(b) * cos(c)
            c = math.atan2(R[i, j], sign * R[i, other])
        else:
            # Gimbal lock: b ≈ 0 or b ≈ pi
            a = math.atan2(sign * R[other, j], R[j, j])
            c = 0.0
    else:
        # Tait-Bryan angles: all three axes distinct
        # R = Ri(a) Rj(b) Rk(c)
        sign = 1.0 if (j - i) % 3 == 1 else -1.0

        sb = np.clip(sign * R[i, k], -1.0, 1.0)
        b = math.asin(sb)

        if abs(math.cos(b)) > 1e-8:
            a = math.atan2(-sign * R[j, k], R[k, k])
            c = math.atan2(-sign * R[i, j], R[i, i])
        else:
            # Gimbal lock
            a = math.atan2(sign * R[k, j], R[j, j])
            c = 0.0

    return (float(a), float(b), float(c))


# ===========================================================================
# Euler <-> Rotation Matrix (convenience, routes through quaternion hub)
# ===========================================================================


def euler_to_rotation_matrix(
    a: float, b: float, c: float, convention: str
) -> np.ndarray:
    """Convert Euler angles to rotation matrix (via quaternion hub, DRY)."""
    if not isinstance(a, (int, float)):
        raise TypeError(f"a must be a number, got {type(a).__name__!r}")
    q = euler_to_quaternion(a, b, c, convention)
    return quaternion_to_rotation_matrix(q)


def rotation_matrix_to_euler(R: Any, convention: str) -> tuple[float, float, float]:
    """Convert rotation matrix to Euler angles (via quaternion hub, DRY)."""
    if not isinstance(convention, str):
        raise TypeError(f"convention must be a str, got {type(convention).__name__!r}")
    R = _validate_rotation_matrix(R)
    q = rotation_matrix_to_quaternion(R)
    return quaternion_to_euler(q, convention)
