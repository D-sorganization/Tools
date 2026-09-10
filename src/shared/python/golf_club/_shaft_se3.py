"""Objective section interpolation for the candidate loaded-shaft element.

Twists are linear-first [translation generator in m, rotation in rad].
Section poses map material axes into one common observer frame. This chart
is kinematics only: it does not establish strain limits, equilibrium or a
dynamic tangent. The principal relative rotation must remain below pi.
"""

from __future__ import annotations

from math import factorial

import numpy as np
from numpy.polynomial.polynomial import polyder, polyval
from scipy.linalg import expm, expm_frechet
from scipy.spatial.transform import Rotation

from ._grip_contracts import finite_array
from ._rotating_body_kernel import cross_matrix
from ._validation import require_finite_float, require_rotation

# A numerical chart guard, not an allowable material rotation/strain claim.
_LOG_BRANCH_MARGIN_RAD = 1e-6
_POLYNOMIAL_NORM_LIMIT = 4.0
_COEFFICIENT_DEGREE = 18


def _coefficient_term(order: int) -> tuple[float, float, float, float]:
    """Taylor coefficients of the four SE(3) Hermite coefficient functions."""
    sign = (-1.0) ** order
    return (
        sign * (1 - order) / factorial(2 * order + 2),
        sign * (1 - order) / factorial(2 * order + 3),
        sign * (1 + order) / factorial(2 * order + 4),
        sign * (1 + order) / factorial(2 * order + 5),
    )


_COEFFICIENTS = np.array(
    [_coefficient_term(order) for order in range(_COEFFICIENT_DEGREE + 1)]
)
_COEFFICIENTS_AND_RATES = np.column_stack(
    (
        _COEFFICIENTS,
        np.vstack((polyder(_COEFFICIENTS), np.zeros(4))),
    )
)
_COEFFICIENTS_AND_RATES.setflags(write=False)


def _rotation_chart(vector: np.ndarray) -> None:
    if np.linalg.norm(vector) >= np.pi - _LOG_BRANCH_MARGIN_RAD:
        raise ValueError("rotation reaches the principal-logarithm branch boundary")


def _phi1_generator(matrix: np.ndarray) -> np.ndarray:
    size = len(matrix)
    generator = np.zeros((2 * size, 2 * size))
    generator[:size, :size] = matrix
    generator[:size, size:] = np.eye(size)
    return generator


def _left_rotation_jacobian(vector: np.ndarray) -> np.ndarray:
    """Integrate Exp(t*[rotation]x), retaining the exact zero-angle limit.

    The upper block of Exp([[A, I], [0, 0]]) is phi_1(A). Using SciPy's
    matrix exponential avoids divisions by small angles and a dead zone.
    """
    generator = _phi1_generator(cross_matrix(vector))
    return np.asarray(expm(generator)[:3, 3:])


def twist_ad(value: object) -> np.ndarray:
    """Return the Lie bracket matrix for a finite linear-first twist."""
    twist = finite_array(value, (6,), "section twist")
    angular = cross_matrix(twist[3:])
    result = np.zeros((6, 6))
    result[:3, :3] = result[3:, 3:] = angular
    result[:3, 3:] = cross_matrix(twist[:3])
    return result


def right_jacobian(value: object) -> np.ndarray:
    """Map exponential-coordinate rates to material twists: phi_1(-ad(q))."""
    generator = _phi1_generator(-twist_ad(value))
    return finite_array(expm(generator)[:6, 6:], (6, 6), "right Jacobian")


def right_jacobian_derivative(value: object, direction: object) -> np.ndarray:
    """Differentiate the right Jacobian without differencing or angle cutoffs.

    A zero direction has exactly zero derivative after validating both inputs.
    Otherwise share the bounded SE(3) evaluator and its general fallback.
    """
    twist = finite_array(value, (6,), "section twist")
    variation = finite_array(direction, (6,), "section twist direction")
    if not np.any(variation):
        return np.zeros((6, 6))
    result = _jacobian_pair(twist, variation)[1]
    return finite_array(result, (6, 6), "right Jacobian derivative")


def _rigid_pose(value: object) -> np.ndarray:
    pose = finite_array(value, (4, 4), "section pose")
    require_rotation(pose[:3, :3])
    if not np.array_equal(pose[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("section pose must have homogeneous bottom row [0, 0, 0, 1]")
    return pose


def exp_twist(value: object) -> np.ndarray:
    """Map a finite linear-first local twist into a fresh rigid transform.

    Rotations at or within 1e-6 rad of pi are outside this local chart.
    No coercion from strings, booleans or complex values is permitted.
    """
    twist = finite_array(value, (6,), "section twist")
    angular = twist[3:]
    _rotation_chart(angular)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_rotvec(angular).as_matrix()
    with np.errstate(over="ignore", invalid="ignore"):
        pose[:3, 3] = _left_rotation_jacobian(angular) @ twist[:3]
    return _rigid_pose(pose)


def log_pose(value: object) -> np.ndarray:
    """Return the principal linear-first twist of a proper rigid transform.

    The rotation branch margin is explicit. Translation is the exponential
    generator, not the endpoint position. Input arrays are never modified.
    """
    pose = _rigid_pose(value)
    angular = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    _rotation_chart(angular)
    linear = np.linalg.solve(_left_rotation_jacobian(angular), pose[:3, 3])
    return finite_array(np.r_[linear, angular], (6,), "section logarithm")


def _material_fraction(value: object) -> float:
    coordinate = require_finite_float(value, "material fraction")
    if not 0.0 <= coordinate <= 1.0:
        raise ValueError("material fraction must lie in [0, 1]")
    return float(coordinate)


def _relative_maps(relative: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return relative-log rates using Jr^-1 - Jl^-1 = ad(relative).

    With A=ad(relative), phi1(-A)=exp(-A) phi1(A), so the inverse
    difference is phi1(A)^-1 (exp(A)-I)=A, including singular A by
    analytic continuation. No inversion of A or angular cutoff is needed.
    """
    right = np.linalg.solve(right_jacobian(relative), np.eye(6))
    left = right - twist_ad(relative)
    return np.hstack((-left, right)), left, right


def _velocity_inputs(
    relative: object, fraction: object
) -> tuple[np.ndarray, float, np.ndarray]:
    twist = finite_array(relative, (6,), "relative section twist")
    _rotation_chart(twist[3:])
    coordinate = _material_fraction(fraction)
    inverse = np.linalg.solve(right_jacobian(twist), np.eye(6))
    return twist, coordinate, inverse


def section_velocity_map(relative: object, fraction: object) -> np.ndarray:
    """Map twelve nodal material velocities to one interpolated material twist.

    For d=Log(H_left^-1 H_right) and alpha in [0,1], Q=[I-B,B],
    B=alpha Jr(alpha*d) Jr(d)^-1. The same map transfers virtual work.
    Principal-chart limits are numerical, not a material-domain qualification.
    """
    twist, coordinate, inverse = _velocity_inputs(relative, fraction)
    right = coordinate * right_jacobian(coordinate * twist) @ inverse
    return finite_array(
        np.hstack((np.eye(6) - right, right)), (6, 12), "section velocity map"
    )


def section_velocity_map_derivative(
    relative: object, fraction: object, direction: object
) -> np.ndarray:
    """Differentiate Q with respect to its relative logarithm in one direction.

    Direction is d_dot or a relative-log variation, not an unconverted nodal
    material velocity. Analytic matrix-function derivatives retain zero and
    tiny rotations; no finite differences or imposed symmetry are used.
    """
    return section_velocity_kinematics(relative, fraction, direction)[1]


def _weighted_powers(
    coefficients: np.ndarray, powers: tuple[np.ndarray, ...]
) -> np.ndarray:
    """Combine the four terms of the same matrix polynomial or its variation."""
    result: np.ndarray = (
        coefficients[0] * powers[0]
        + coefficients[1] * powers[1]
        + coefficients[2] * powers[2]
        + coefficients[3] * powers[3]
    )
    return result


def _polynomial_jacobian_pair(
    twist: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Use the SE(3) degree-four Hermite form with stable scalar coefficients.

    The matrix minimal polynomial divides x*(x*x+theta*theta)**2. The scalar
    coefficient series and their derivatives are evaluated through degree 18
    in theta**2, without small-angle divisions. Qualified numerical domain:
    ||ad(twist)||_inf <= 4 in the declared SI coordinates, theta <= pi.
    Return None outside it; no physical/chart limit is relaxed. Inputs are
    finite real six-vectors; outputs are fresh and inputs are not modified.
    """
    matrix, change = -twist_ad(twist), -twist_ad(direction)
    angle_squared = float(np.dot(twist[3:], twist[3:]))
    if (
        np.linalg.norm(matrix, ord=np.inf) > _POLYNOMIAL_NORM_LIMIT
        or angle_squared > np.pi**2
    ):
        return None
    values = polyval(angle_squared, _COEFFICIENTS_AND_RATES)
    coefficients = values[:4]
    rates = values[4:] * (2 * np.dot(twist[3:], direction[3:]))
    square = matrix @ matrix
    cube = square @ matrix
    fourth = cube @ matrix
    square_rate = change @ matrix + matrix @ change
    cube_rate = square_rate @ matrix + square @ change
    fourth_rate = cube_rate @ matrix + cube @ change
    powers = matrix, square, cube, fourth
    changes = change, square_rate, cube_rate, fourth_rate
    return (
        np.eye(6) + _weighted_powers(coefficients, powers),
        _weighted_powers(rates, powers) + _weighted_powers(coefficients, changes),
    )


def _jacobian_pair(
    twist: np.ndarray, direction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Use bounded SE(3) evaluation or the original general Frechet routine."""
    polynomial = _polynomial_jacobian_pair(twist, direction)
    if polynomial is not None:
        return polynomial
    generator = _phi1_generator(-twist_ad(twist))
    variation = np.zeros_like(generator)
    variation[:6, :6] = -twist_ad(direction)
    exponential, derivative = expm_frechet(generator, variation, compute_expm=True)
    return exponential[:6, 6:], derivative[:6, 6:]


class _SectionVelocityKinematics:
    """Own one section state and lazily reuse its full Frechet pair.

    This object belongs to one evaluation, never a global or cross-state cache.
    Relative pose and direction are copied; each fraction returns fresh arrays.
    Endpoint-only quadrature still needs no exponential.
    """

    def __init__(self, relative: object, direction: object) -> None:
        self._twist = finite_array(relative, (6,), "relative section twist")
        _rotation_chart(self._twist[3:])
        self._direction = finite_array(direction, (6,), "relative twist direction")
        self._twist.setflags(write=False)
        self._direction.setflags(write=False)
        self._prepared: tuple[np.ndarray, np.ndarray] | None = None

    def _matrices(self) -> tuple[np.ndarray, np.ndarray]:
        if self._prepared is None:
            full, full_rate = _jacobian_pair(self._twist, self._direction)
            inverse = np.linalg.solve(full, np.eye(6))
            inverse.setflags(write=False)
            full_rate.setflags(write=False)
            self._prepared = inverse, full_rate
        return self._prepared

    def at(self, fraction: object) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one validated material fraction with the original algebra."""
        coordinate = _material_fraction(fraction)
        if coordinate in (0.0, 1.0):
            mapping = np.zeros((6, 12))
            start = 6 * int(coordinate)
            mapping[:, start : start + 6] = np.eye(6)
            return mapping, np.zeros((6, 12))
        inverse, full_rate = self._matrices()
        part, part_rate = _jacobian_pair(
            coordinate * self._twist, coordinate * self._direction
        )
        right = coordinate * part @ inverse
        derivative = coordinate * (part_rate - part @ inverse @ full_rate) @ inverse
        mapping = np.hstack((np.eye(6) - right, right))
        rate = np.hstack((-derivative, derivative))
        return (
            finite_array(mapping, (6, 12), "section velocity map"),
            finite_array(rate, (6, 12), "section velocity-map derivative"),
        )


def section_velocity_kinematics(
    relative: object, fraction: object, direction: object
) -> tuple[np.ndarray, np.ndarray]:
    """Return Q and its directional derivative using two Frechet evaluations.

    This is the same interpolation as the individual map/derivative routines.
    Exact endpoint identities avoid matrix exponentials. All input and output
    domains remain checked, including endpoint directions and chart limits.
    No cross-evaluation cache, angle cutoff, finite difference or coefficient
    change occurs. Inertia quadrature reuses the same private kernel across
    fractions within one evaluation.
    """
    return _SectionVelocityKinematics(relative, direction).at(fraction)


def interpolate_pose(left: object, right: object, fraction: object) -> np.ndarray:
    """Evaluate H_left Exp(fraction Log(H_left^-1 H_right)).

    Both poses must use the same observer frame and fraction must be a finite
    real number in [0, 1]. Constant material strain yields a circular centerline
    under pure bending; a common rigid motion leaves relative strain unchanged.
    """
    coordinate = _material_fraction(fraction)
    left_pose, right_pose = _rigid_pose(left), _rigid_pose(right)
    relative = np.linalg.solve(left_pose, right_pose)
    return _rigid_pose(left_pose @ exp_twist(coordinate * log_pose(relative)))


__all__ = ()
