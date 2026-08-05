"""Validated value types for six-DOF contact-interval simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from shared.python.swing_sim.impact import GOLF_BALL_MASS_KG, PostImpactState

from .contact import KelvinVoigtContactLaw


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


class BoundaryKind(Enum):
    """Grip/shaft idealization applied at the attachment point."""

    FREE = "free"
    PINNED = "pinned"
    TORSIONAL_GRIP = "torsional_grip"


@dataclass(frozen=True)
class ClubRigidBody:
    """Rigid-club properties in a body-fixed frame, all in SI units."""

    mass_kg: float
    inertia_body_kg_m2: np.ndarray
    cg_to_contact_body_m: np.ndarray
    cg_to_attachment_body_m: np.ndarray
    face_normal_body: np.ndarray

    def __post_init__(self) -> None:
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be finite and positive")
        inertia = _matrix(self.inertia_body_kg_m2, "inertia_body_kg_m2")
        if not np.allclose(inertia, inertia.T, atol=1e-12):
            raise ValueError("inertia_body_kg_m2 must be symmetric")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ValueError("inertia_body_kg_m2 must be positive definite")
        _vector(self.cg_to_contact_body_m, "cg_to_contact_body_m")
        attachment = _vector(self.cg_to_attachment_body_m, "cg_to_attachment_body_m")
        if np.linalg.norm(attachment) <= 1e-9:
            raise ValueError("cg_to_attachment_body_m must be non-zero")
        normal = _vector(self.face_normal_body, "face_normal_body")
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-8):
            raise ValueError("face_normal_body must be a unit vector")

    @property
    def shaft_axis_body(self) -> np.ndarray:
        """Unit axis from club CG toward the grip attachment."""
        vector = np.asarray(self.cg_to_attachment_body_m, dtype=float)
        return vector / np.linalg.norm(vector)


@dataclass
class ImpactIntervalInitialState:
    """Complete club and ball state at or immediately before first contact."""

    club_position_m: np.ndarray
    club_orientation: np.ndarray
    club_velocity_mps: np.ndarray
    club_angular_velocity_rad_s: np.ndarray
    ball_position_m: np.ndarray
    ball_velocity_mps: np.ndarray
    ball_angular_velocity_rad_s: np.ndarray

    def validate(self) -> None:
        """Validate shapes, finiteness, and the SO(3) orientation contract."""
        for name in (
            "club_position_m",
            "club_velocity_mps",
            "club_angular_velocity_rad_s",
            "ball_position_m",
            "ball_velocity_mps",
            "ball_angular_velocity_rad_s",
        ):
            _vector(getattr(self, name), name)
        rotation = _matrix(self.club_orientation, "club_orientation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
            raise ValueError("club_orientation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("club_orientation must be a proper rotation")


@dataclass(frozen=True)
class ImpactIntervalConfig:
    """Numerical, contact, friction, and boundary configuration."""

    contact_law: KelvinVoigtContactLaw
    time_step_s: float = 1.0e-7
    maximum_time_s: float = 2.0e-3
    friction_coefficient: float = 0.4
    friction_regularization_mps: float = 1.0e-3
    boundary: BoundaryKind = BoundaryKind.FREE
    torsional_stiffness_n_m_per_rad: float = 0.0
    torsional_damping_n_m_s_per_rad: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.contact_law, KelvinVoigtContactLaw):
            raise TypeError("contact_law must be a KelvinVoigtContactLaw")
        if not isinstance(self.boundary, BoundaryKind):
            raise TypeError("boundary must be a BoundaryKind")
        positive = ("time_step_s", "maximum_time_s", "friction_regularization_mps")
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "friction_coefficient",
            "torsional_stiffness_n_m_per_rad",
            "torsional_damping_n_m_s_per_rad",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.time_step_s >= self.maximum_time_s:
            raise ValueError("time_step_s must be smaller than maximum_time_s")
        if self.boundary is not BoundaryKind.TORSIONAL_GRIP and (
            self.torsional_stiffness_n_m_per_rad > 0.0
            or self.torsional_damping_n_m_s_per_rad > 0.0
        ):
            raise ValueError("torsional parameters require TORSIONAL_GRIP boundary")


@dataclass(frozen=True)
class ImpactIntervalAudit:
    """Scientific reconciliation metrics for one interval solve."""

    initial_kinetic_energy_j: float
    final_kinetic_energy_j: float
    dissipated_energy_j: float
    dashpot_and_friction_dissipation_j: float
    unilateral_release_energy_j: float
    boundary_stored_energy_j: float
    energy_residual_j: float
    integrated_normal_impulse_n_s: float
    integrated_friction_impulse_n_s: float
    linear_momentum_residual_n_s: float


@dataclass(frozen=True)
class ImpactIntervalSample:
    """One nearest-neighbor sample from an interval history."""

    time_s: float
    club_position_m: np.ndarray
    club_orientation: np.ndarray
    club_velocity_mps: np.ndarray
    club_angular_velocity_rad_s: np.ndarray
    ball_position_m: np.ndarray
    ball_velocity_mps: np.ndarray
    ball_angular_velocity_rad_s: np.ndarray
    normal_force_n: float
    compression_m: float
    face_angle_deg: float
    dynamic_loft_deg: float
    twist_angle_rad: float


@dataclass(frozen=True)
class ImpactIntervalResult:
    """Queryable, traceable state history through one contact interval."""

    time_s: np.ndarray
    club_position_m: np.ndarray
    club_orientation: np.ndarray
    club_velocity_mps: np.ndarray
    club_angular_velocity_rad_s: np.ndarray
    ball_position_m: np.ndarray
    ball_velocity_mps: np.ndarray
    ball_angular_velocity_rad_s: np.ndarray
    attachment_position_m: np.ndarray
    contact_point_position_m: np.ndarray
    contact_normal: np.ndarray
    normal_force_n: np.ndarray
    friction_force_n: np.ndarray
    compression_m: np.ndarray
    face_angle_deg: np.ndarray
    dynamic_loft_deg: np.ndarray
    twist_angle_rad: np.ndarray
    contact_duration_s: float
    did_contact: bool
    audit: ImpactIntervalAudit

    def channel(self, name: str) -> np.ndarray:
        """Return a named history channel without exposing implementation modules."""
        channels = {
            field: getattr(self, field)
            for field in (
                "time_s",
                "club_position_m",
                "club_orientation",
                "club_velocity_mps",
                "club_angular_velocity_rad_s",
                "ball_position_m",
                "ball_velocity_mps",
                "ball_angular_velocity_rad_s",
                "attachment_position_m",
                "contact_point_position_m",
                "contact_normal",
                "normal_force_n",
                "friction_force_n",
                "compression_m",
                "face_angle_deg",
                "dynamic_loft_deg",
                "twist_angle_rad",
            )
        }
        try:
            return channels[name]
        except KeyError as exc:
            raise ValueError(f"Unknown impact-interval channel: {name}") from exc

    def at_time(self, time_s: float) -> ImpactIntervalSample:
        """Return the stored sample nearest ``time_s`` (clamped to history)."""
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")
        index = int(np.argmin(np.abs(self.time_s - time_s)))
        return ImpactIntervalSample(
            time_s=float(self.time_s[index]),
            club_position_m=self.club_position_m[index].copy(),
            club_orientation=self.club_orientation[index].copy(),
            club_velocity_mps=self.club_velocity_mps[index].copy(),
            club_angular_velocity_rad_s=self.club_angular_velocity_rad_s[index].copy(),
            ball_position_m=self.ball_position_m[index].copy(),
            ball_velocity_mps=self.ball_velocity_mps[index].copy(),
            ball_angular_velocity_rad_s=self.ball_angular_velocity_rad_s[index].copy(),
            normal_force_n=float(self.normal_force_n[index]),
            compression_m=float(self.compression_m[index]),
            face_angle_deg=float(self.face_angle_deg[index]),
            dynamic_loft_deg=float(self.dynamic_loft_deg[index]),
            twist_angle_rad=float(self.twist_angle_rad[index]),
        )

    def to_post_impact_state(self) -> PostImpactState:
        """Adapt the final interval state to the established impact façade."""
        ball_pre_ke = (
            0.5
            * GOLF_BALL_MASS_KG
            * float(np.dot(self.ball_velocity_mps[0], self.ball_velocity_mps[0]))
        )
        ball_post_ke = (
            0.5
            * GOLF_BALL_MASS_KG
            * float(np.dot(self.ball_velocity_mps[-1], self.ball_velocity_mps[-1]))
        )
        return PostImpactState(
            ball_velocity=self.ball_velocity_mps[-1].copy(),
            ball_angular_velocity=self.ball_angular_velocity_rad_s[-1].copy(),
            clubhead_velocity=self.club_velocity_mps[-1].copy(),
            clubhead_angular_velocity=self.club_angular_velocity_rad_s[-1].copy(),
            contact_duration=self.contact_duration_s,
            energy_transfer=ball_post_ke - ball_pre_ke,
            impact_location=np.zeros(2),
        )


__all__ = [
    "BoundaryKind",
    "ClubRigidBody",
    "ImpactIntervalAudit",
    "ImpactIntervalConfig",
    "ImpactIntervalInitialState",
    "ImpactIntervalResult",
    "ImpactIntervalSample",
]
