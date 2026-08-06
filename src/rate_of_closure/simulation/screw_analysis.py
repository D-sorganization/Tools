"""Typed instantaneous screw-motion analysis for clubs and articulated joints.

The app-frame convention is ``x`` toward the target, ``y`` up, and ``z``
right of target.  A club twist stores angular velocity followed by the linear
velocity of the explicitly supplied reference point.  This differs from a
spatial twist whose linear term is defined at the world origin, so the
reference point is a mandatory part of this API.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from rate_of_closure._contracts import ensure, require

__all__ = [
    "JointMotionSeries",
    "MotionKind",
    "MotionProjection",
    "ScrewGlyph",
    "ScrewMotion",
    "analyze_joint_motion",
    "analyze_twist",
    "build_screw_glyph",
    "project_motion",
]

_MOTION_EPS = 1e-10
_GLYPH_POINTS = 96
_DEFAULT_DIRECTIONS = MappingProxyType(
    {
        "target": np.array([1.0, 0.0, 0.0]),
        "vertical": np.array([0.0, 1.0, 0.0]),
        "lateral": np.array([0.0, 0.0, 1.0]),
    }
)


class MotionKind(StrEnum):
    """Truthful geometric state of an instantaneous rigid-body motion."""

    FINITE = "finite"
    TRANSLATION = "translation"
    STATIONARY = "stationary"


@dataclass(frozen=True)
class ScrewMotion:
    """One rigid body's instantaneous screw decomposition in the app frame."""

    kind: MotionKind
    reference_point_m: np.ndarray
    reference_velocity_m_s: np.ndarray
    angular_velocity_rad_s: np.ndarray
    axis_direction: np.ndarray
    axis_point_m: np.ndarray | None
    pitch_m_rad: float | None
    angular_rate_rad_s: float
    axial_speed_m_s: float
    radius_m: float | None
    orbital_velocity_m_s: np.ndarray
    axial_velocity_m_s: np.ndarray
    reconstruction_residual_m_s: float


@dataclass(frozen=True)
class MotionProjection:
    """Signed velocity breakdown along one named unit direction."""

    direction: np.ndarray
    total_m_s: float
    orbital_m_s: float
    axial_m_s: float


@dataclass(frozen=True)
class ScrewGlyph:
    """Renderer-neutral engineering glyph for one finite screw axis."""

    axis_line_m: np.ndarray
    helix_m: np.ndarray
    radius_line_m: np.ndarray
    handedness: int


@dataclass(frozen=True)
class JointMotionSeries:
    """Per-joint revolute screw contributions to the distal endpoint."""

    joint_ids: tuple[str, ...]
    axis_points_m: np.ndarray
    angular_velocity_rad_s: np.ndarray
    contribution_velocity_m_s: np.ndarray
    endpoint_velocity_m_s: np.ndarray
    reconstruction_residual_m_s: np.ndarray


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    """Return a finite three-vector after enforcing the public contract."""
    vector = np.asarray(value, dtype=float)
    require(vector.shape == (3,), f"{name} must have shape (3,)", vector.shape)
    require(bool(np.all(np.isfinite(vector))), f"{name} must be finite", vector)
    return vector


def _finite_motion(
    omega: np.ndarray, velocity: np.ndarray, reference: np.ndarray
) -> ScrewMotion:
    """Decompose a nonzero-angular-velocity reference-point twist."""
    rate_squared = float(omega @ omega)
    rate = math.sqrt(rate_squared)
    direction = omega / rate
    pitch = float(omega @ velocity / rate_squared)
    axis_point = reference + np.cross(omega, velocity) / rate_squared
    axial = pitch * omega
    orbital = velocity - axial
    radius = float(np.linalg.norm(reference - axis_point))
    residual = float(np.linalg.norm(orbital + axial - velocity))
    ensure(residual <= 1e-9, "screw components must reconstruct velocity", residual)
    return ScrewMotion(
        MotionKind.FINITE,
        reference,
        velocity,
        omega,
        direction,
        axis_point,
        pitch,
        rate,
        pitch * rate,
        radius,
        orbital,
        axial,
        residual,
    )


def analyze_twist(twist: np.ndarray, reference_point_m: np.ndarray) -> ScrewMotion:
    """Decompose ``[omega, velocity_at_reference]`` into screw motion.

    Pure translation has its screw axis at infinity and stationary motion has
    no axis.  Both therefore return ``axis_point_m=None`` rather than inventing
    finite display geometry.
    """
    values = np.asarray(twist, dtype=float)
    require(values.shape == (6,), "twist must have shape (6,)", values.shape)
    require(bool(np.all(np.isfinite(values))), "twist must be finite", values)
    reference = _vector3(reference_point_m, "reference_point_m")
    omega, velocity = values[:3], values[3:]
    if float(np.linalg.norm(omega)) > _MOTION_EPS:
        return _finite_motion(omega, velocity, reference)
    speed = float(np.linalg.norm(velocity))
    kind = MotionKind.TRANSLATION if speed > _MOTION_EPS else MotionKind.STATIONARY
    direction = velocity / speed if speed > _MOTION_EPS else np.zeros(3)
    return ScrewMotion(
        kind,
        reference,
        velocity,
        omega,
        direction,
        None,
        None,
        0.0,
        speed if kind is MotionKind.TRANSLATION else 0.0,
        None,
        np.zeros(3),
        velocity.copy(),
        0.0,
    )


def _unit_direction(value: np.ndarray, name: str) -> np.ndarray:
    """Return a validated unit direction for a signed projection."""
    direction = _vector3(value, name)
    magnitude = float(np.linalg.norm(direction))
    require(magnitude > _MOTION_EPS, f"{name} must be nonzero", direction)
    return direction / magnitude


def project_motion(
    motion: ScrewMotion,
    directions: Mapping[str, np.ndarray] | None = None,
) -> dict[str, MotionProjection]:
    """Project total, orbital, and axial velocity onto named directions."""
    require(isinstance(motion, ScrewMotion), "motion must be a ScrewMotion")
    selected = directions or _DEFAULT_DIRECTIONS
    result: dict[str, MotionProjection] = {}
    for name, raw_direction in selected.items():
        require(bool(name.strip()), "projection names must be nonempty", name)
        direction = _unit_direction(raw_direction, f"direction[{name}]")
        result[name] = MotionProjection(
            direction,
            float(motion.reference_velocity_m_s @ direction),
            float(motion.orbital_velocity_m_s @ direction),
            float(motion.axial_velocity_m_s @ direction),
        )
    return result


def _orthogonal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic right-handed basis normal to ``axis``."""
    seed = np.eye(3)[int(np.argmin(np.abs(axis)))]
    first = np.cross(axis, seed)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _dominant_sign(vector: np.ndarray) -> int:
    """Encode angular direction independently of floating-point near-zero noise."""
    value = float(vector[int(np.argmax(np.abs(vector)))])
    return 1 if value >= 0.0 else -1


def build_screw_glyph(motion: ScrewMotion, scene_extent_m: float) -> ScrewGlyph | None:
    """Build bounded axis, helix, and reference-radius geometry.

    Axis length grows monotonically with angular rate but is capped by the
    current swing-scale extent, preventing extreme rates or pitch from changing
    plot bounds.  Helix handedness follows the dominant signed angular component.
    """
    require(
        math.isfinite(scene_extent_m) and scene_extent_m > 0.0,
        "scene_extent_m must be finite and > 0",
        scene_extent_m,
    )
    if motion.kind is not MotionKind.FINITE or motion.axis_point_m is None:
        return None
    growth = 0.55 + 0.35 * math.tanh(motion.angular_rate_rad_s / 10.0)
    half_length = scene_extent_m * growth
    axis = motion.axis_direction
    point = motion.axis_point_m
    axis_line = np.vstack([point - half_length * axis, point + half_length * axis])
    first, second = _orthogonal_basis(axis)
    handedness = _dominant_sign(motion.angular_velocity_rad_s)
    phase = np.linspace(-2.0 * math.pi, 2.0 * math.pi, _GLYPH_POINTS)
    axial = np.linspace(-0.82 * half_length, 0.82 * half_length, _GLYPH_POINTS)
    radius = scene_extent_m * 0.055
    helix = (
        point
        + np.outer(axial, axis)
        + radius * np.outer(np.cos(phase), first)
        + handedness * radius * np.outer(np.sin(phase), second)
    )
    radius_line = np.vstack([point, motion.reference_point_m])
    return ScrewGlyph(axis_line, helix, radius_line, handedness)


def _joint_inputs(
    times_s: np.ndarray, joint_positions_m: np.ndarray, joint_ids: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize sampled articulated geometry."""
    times = np.asarray(times_s, dtype=float)
    points = np.asarray(joint_positions_m, dtype=float)
    require(times.ndim == 1 and len(times) >= 3, "times_s must contain >= 3 samples")
    require(bool(np.all(np.isfinite(times))), "times_s must be finite")
    require(bool(np.all(np.diff(times) > 0.0)), "times_s must be strictly increasing")
    require(
        points.shape == (len(times), len(joint_ids) + 1, 3),
        "joint_positions_m must have shape (N, len(joint_ids)+1, 3)",
        points.shape,
    )
    require(bool(np.all(np.isfinite(points))), "joint_positions_m must be finite")
    require(len(set(joint_ids)) == len(joint_ids), "joint_ids must be unique")
    return times, points


def analyze_joint_motion(
    times_s: np.ndarray,
    joint_positions_m: np.ndarray,
    joint_ids: tuple[str, ...],
) -> JointMotionSeries:
    """Reconstruct planar revolute-joint contributions from sampled link geometry.

    Each joint's angular velocity is the relative angular velocity between its
    distal and proximal links.  Contributions are evaluated at the distal
    endpoint.  The returned residual quantifies disagreement with the numerical
    derivative of that endpoint and must remain visible to consumers.
    """
    times, points = _joint_inputs(times_s, joint_positions_m, joint_ids)
    segments = points[:, 1:, :] - points[:, :-1, :]
    segment_rates = np.gradient(segments, times, axis=0, edge_order=2)
    lengths_squared = np.einsum("nji,nji->nj", segments, segments)
    require(bool(np.all(lengths_squared > _MOTION_EPS)), "segments must be nonzero")
    absolute_omega = np.cross(segments, segment_rates) / lengths_squared[..., None]
    relative_omega = absolute_omega.copy()
    relative_omega[:, 1:, :] -= absolute_omega[:, :-1, :]
    endpoint = points[:, -1, :]
    arms = endpoint[:, None, :] - points[:, :-1, :]
    contributions = np.cross(relative_omega, arms)
    endpoint_velocity = np.gradient(endpoint, times, axis=0, edge_order=2)
    residual = np.linalg.norm(contributions.sum(axis=1) - endpoint_velocity, axis=1)
    return JointMotionSeries(
        tuple(joint_ids),
        points[:, :-1, :].copy(),
        relative_omega,
        contributions,
        endpoint_velocity,
        residual,
    )
