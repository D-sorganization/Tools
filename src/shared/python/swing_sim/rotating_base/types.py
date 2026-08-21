"""Typed state and result contracts for rotating-base dynamics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._numeric import N_COORDINATES, FloatArray, _finite_vector


@dataclass(frozen=True, slots=True)
class RotatingBaseParams:
    """Physical parameters for the qualified reduced mechanism tier."""

    torso_inertia_kg_m2: float = 4.8
    torso_stiffness_nm_rad: float = 0.0
    torso_damping_nms_rad: float = 0.35
    lead_shoulder_offset_m: tuple[float, float] = (0.19, 0.0)
    trail_shoulder_offset_m: tuple[float, float] = (-0.19, 0.0)
    arm_length_m: float = 0.62
    arm_mass_kg: float = 3.1
    arm_inertia_kg_m2: float = 0.0993
    arm_damping_nms_rad: float = 0.10
    proximal_club_length_m: float = 0.46
    distal_club_length_m: float = 0.54
    proximal_club_mass_kg: float = 0.18
    distal_club_mass_kg: float = 0.27
    proximal_club_inertia_kg_m2: float = 0.003174
    distal_club_inertia_kg_m2: float = 0.006561
    shaft_stiffness_nm_rad: float = 80.0
    shaft_damping_nms_rad: float = 0.6
    lead_grip_offset_m: float = 0.065
    trail_grip_offset_m: float = -0.065
    gravity_m_s2: float = 9.80665
    rank_tolerance: float = 1e-10
    kkt_tolerance: float = 1e-8

    @classmethod
    def publication_default(cls) -> RotatingBaseParams:
        """Return the deterministic publication parameter set."""
        return cls()

    def __post_init__(self) -> None:
        positive = (
            "torso_inertia_kg_m2",
            "arm_length_m",
            "arm_mass_kg",
            "arm_inertia_kg_m2",
            "proximal_club_length_m",
            "distal_club_length_m",
            "proximal_club_mass_kg",
            "distal_club_mass_kg",
            "proximal_club_inertia_kg_m2",
            "distal_club_inertia_kg_m2",
            "shaft_stiffness_nm_rad",
            "rank_tolerance",
            "kkt_tolerance",
        )
        nonnegative = (
            "torso_stiffness_nm_rad",
            "torso_damping_nms_rad",
            "arm_damping_nms_rad",
            "shaft_damping_nms_rad",
            "gravity_m_s2",
        )
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("lead_shoulder_offset_m", "trail_shoulder_offset_m"):
            _finite_vector(name, getattr(self, name), (2,))
        grips = np.array([self.lead_grip_offset_m, self.trail_grip_offset_m])
        if not np.all(np.isfinite(grips)):
            raise ValueError("grip offsets must be finite")


@dataclass(frozen=True, slots=True)
class RotatingBaseConfig:
    """Integration and projection contract."""

    duration_s: float
    step_s: float
    projection_tolerance_m: float = 1e-11
    velocity_tolerance_m_s: float = 1e-10
    maximum_projection_iterations: int = 16

    def __post_init__(self) -> None:
        for name in (
            "duration_s",
            "step_s",
            "projection_tolerance_m",
            "velocity_tolerance_m_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_projection_iterations < 1:
            raise ValueError("maximum_projection_iterations must be positive")
        intervals = self.duration_s / self.step_s
        if not np.isclose(intervals, round(intervals), atol=1e-10):
            raise ValueError("duration_s must be an integer multiple of step_s")

    @property
    def interval_count(self) -> int:
        return int(round(self.duration_s / self.step_s))


@dataclass(frozen=True, slots=True)
class RotatingBaseState:
    """One finite generalized state."""

    q: FloatArray
    qdot: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", _finite_vector("q", self.q, (N_COORDINATES,)))
        object.__setattr__(
            self, "qdot", _finite_vector("qdot", self.qdot, (N_COORDINATES,))
        )


@dataclass(frozen=True, slots=True)
class TorsoTwoHandControl:
    """Torso, bilateral arm, and bilateral wrist generalized commands."""

    torso_nm: float = 0.0
    lead_arm_nm: float = 0.0
    trail_arm_nm: float = 0.0
    lead_wrist_nm: float = 0.0
    trail_wrist_nm: float = 0.0

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(tuple(self.as_array()))):
            raise ValueError("control values must be finite")

    def as_array(self) -> FloatArray:
        return np.array(
            [
                self.torso_nm,
                self.lead_arm_nm,
                self.trail_arm_nm,
                self.lead_wrist_nm,
                self.trail_wrist_nm,
            ],
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class DynamicsSolution:
    """One full-rank constrained dynamics solution."""

    qddot: FloatArray
    multipliers_n: FloatArray
    force_on_hands_n: FloatArray
    force_on_club_n: FloatArray
    force_generated_couple_nm: float
    constraint_rank: int
    kkt_residual_norm: float
    acceleration_constraint_residual_norm: float


@dataclass(frozen=True, slots=True)
class RotatingBaseTrace:
    """Qualified trajectory and its force, power, and energy ledgers."""

    time: FloatArray
    q: FloatArray
    qdot: FloatArray
    qddot: FloatArray
    controls: tuple[TorsoTwoHandControl, ...]
    force_on_club_n: FloatArray
    force_generated_couple_nm: FloatArray
    contact_power_on_club_w: FloatArray
    contact_power_identity_residual_w: FloatArray
    clubhead_velocity_m_s: FloatArray
    clubhead_speed_m_s: FloatArray
    distal_segment_kinetic_energy_j: FloatArray
    mechanical_energy_j: FloatArray
    control_power_w: FloatArray
    dissipation_power_w: FloatArray
    position_constraint_norm_m: FloatArray
    velocity_constraint_norm_m_s: FloatArray
    projection_energy_change_j: FloatArray
    work_energy_closure_j: float
    model_tier: str = "planar_rotating_base_two_hand_compliant_club"


ControlLaw = Callable[[float, RotatingBaseState], TorsoTwoHandControl]
