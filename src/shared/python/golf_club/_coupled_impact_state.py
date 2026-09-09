"""Immutable state and energy records shared by the IA-T2 solver and audit API."""

from __future__ import annotations

from dataclasses import dataclass

from ._validation import require_finite_float


@dataclass(frozen=True)
class CoupledImpactInitialState:
    """Finite SI state in the fixed-anchor inertial frame at first touch.

    Ball starts at the head position, with velocity head_velocity - head_speed.
    Displacements encode shaft/grip preload; no equilibrium is silently imposed.
    The anchor remains fixed, so external boundary work is zero.
    """

    head_displacement_m: float = 0.0
    grip_displacement_m: float = 0.0
    head_velocity_mps: float = 0.0
    grip_velocity_mps: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(
                self, name, require_finite_float(getattr(self, name), name)
            )


@dataclass(frozen=True)
class CoupledImpactEnergyLedger:
    """Joules in one inertial frame; residual = initial + work - final - loss.

    Final includes all kinetic and spring potential energy. Contact damping,
    clipped-unloading loss, shaft damping and grip damping are disjoint channels.
    Numerical residual is reported independently and never labeled dissipation.
    """

    initial_energy_j: float
    final_energy_j: float
    contact_damping_dissipation_j: float
    contact_cutoff_dissipation_j: float
    shaft_dissipation_j: float
    grip_dissipation_j: float
    external_work_j: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = require_finite_float(getattr(self, name), name)
            if name != "external_work_j" and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)

    @property
    def total_dissipation_j(self) -> float:
        return (
            self.contact_damping_dissipation_j
            + self.contact_cutoff_dissipation_j
            + self.shaft_dissipation_j
            + self.grip_dissipation_j
        )

    @property
    def residual_j(self) -> float:
        return (
            self.initial_energy_j
            + self.external_work_j
            - self.final_energy_j
            - self.total_dissipation_j
        )


@dataclass(frozen=True)
class CoupledImpactAudit:
    """A completed collision; signed velocities use the simulation frame.

    force_release_time_s is the first decreasing raw-force zero; clearance is
    the first decreasing overlap zero. Later force reactivation before clearance
    remains governed by the unilateral law. dt_s is the maximum accepted step.
    """

    ball_velocity_mps: float
    head_velocity_mps: float
    grip_velocity_mps: float
    clearance_time_s: float
    force_release_time_s: float
    peak_contact_force_n: float
    contact_impulse_n_s: float
    terminal_overlap_m: float
    stored_spring_energy_j: float
    energy: CoupledImpactEnergyLedger
