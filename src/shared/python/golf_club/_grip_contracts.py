"""Immutable local-port records for passive six-axis grip impedance."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._validation import require_finite_float, require_identifier

Vector6 = tuple[float, float, float, float, float, float]
Matrix6 = tuple[tuple[float, ...], ...]


def finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Copy strictly real numeric data; refuse coercion from strings/bools."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain real numbers") from error
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers, not booleans or strings")
    if any(
        isinstance(item, (bool, np.bool_))
        for item in np.asarray(value, dtype=object).flat
    ):
        raise TypeError(f"{name} must not contain booleans")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    array = np.array(array, dtype=float, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _finite_norm(matrix: np.ndarray, name: str) -> float:
    """Finite Frobenius/Euclidean norm without squaring tiny or large entries."""
    value = math.hypot(*np.abs(matrix).ravel())
    if not math.isfinite(value):
        raise ValueError(f"{name} is nonfinite")
    return value


def _node_index(value: object) -> int:
    """Validate a nonnegative material-node index without boolean coercion."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("node must be an integer")
    if value < 0:
        raise ValueError("node must be nonnegative")
    return int(value)


def vector6(value: object, name: str) -> Vector6:
    """Return a copied, finite six-vector in declared SI port coordinates."""
    return tuple(float(item) for item in finite_array(value, (6,), name))  # type: ignore[return-value]


def factor6(value: object, name: str) -> Matrix6:
    """Return an immutable finite Gram factor with finite coefficients."""
    array = finite_array(value, (6, 6), name)
    with np.errstate(over="ignore", invalid="ignore"):
        coefficients = array.T @ array
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"{name} must produce finite impedance coefficients")
    return tuple(tuple(float(item) for item in row) for row in array)


@dataclass(frozen=True)
class PassiveGripImpedance:
    """Constant local linear six-axis impedance, passive by construction.

    Order is translations then small rotations. Each coefficient matrix is
    F.T F, where F is its supplied factor. Inertance maps relative acceleration
    to wrench, damping maps velocity to wrench, and stiffness maps displacement
    to wrench. Gram factors support coupled and rank-deficient passive ports
    without silently repairing an active input coefficient matrix.

    These are ideal local impedance coefficients, not inferred hand mass or a
    pressure-to-damping law. Source identification is mandatory; construction
    does not establish experimental calibration or a frequency-validity band.
    """

    frame_id: str
    inertance_factor: Matrix6
    damping_factor: Matrix6
    stiffness_factor: Matrix6
    source_id: str

    def __post_init__(self) -> None:
        for name in ("frame_id", "source_id"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )
        for name in ("inertance_factor", "damping_factor", "stiffness_factor"):
            object.__setattr__(self, name, factor6(getattr(self, name), name))


@dataclass(frozen=True)
class GripPortState:
    """Local relative displacement, velocity and acceleration at a grip port.

    First three entries have units m, m/s, m/s²; last three are small angular
    displacement in rad, rad/s, rad/s². A fixed local linearization is assumed.
    This is not a finite-rotation pose or an absolute moving-hand trajectory.
    """

    frame_id: str
    displacement: Vector6
    velocity: Vector6
    acceleration: Vector6

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "frame_id", require_identifier(self.frame_id, "frame_id")
        )
        for name in ("displacement", "velocity", "acceleration"):
            object.__setattr__(self, name, vector6(getattr(self, name), name))


@dataclass(frozen=True)
class GripPortResponse:
    """Reaction [N, N m], stored energies [J] and instantaneous power [W].

    Input power is positive into the impedance. Reaction acts on the attached
    shaft and has the opposite sign. Residual is input minus stored-energy
    rate minus dissipation; numerical residual is never counted as damping.
    """

    frame_id: str
    source_id: str
    reaction_wrench: Vector6
    inertial_energy_j: float
    elastic_energy_j: float
    dissipated_power_w: float
    input_power_w: float
    stored_energy_rate_w: float

    def __post_init__(self) -> None:
        for name in ("frame_id", "source_id"):
            object.__setattr__(
                self, name, require_identifier(getattr(self, name), name)
            )
        object.__setattr__(
            self, "reaction_wrench", vector6(self.reaction_wrench, "reaction_wrench")
        )
        nonnegative = ("inertial_energy_j", "elastic_energy_j", "dissipated_power_w")
        for name in (*nonnegative, "input_power_w", "stored_energy_rate_w"):
            value = require_finite_float(getattr(self, name), name)
            if name in nonnegative and value < 0:
                raise ValueError(f"{name} must be >= 0")
            object.__setattr__(self, name, value)
        require_finite_float(self.power_residual_w, "power_residual_w")

    @property
    def power_residual_w(self) -> float:
        """Compute numerical closure separately from the dissipated power."""
        return self.input_power_w - self.stored_energy_rate_w - self.dissipated_power_w


__all__: list[str] = []
