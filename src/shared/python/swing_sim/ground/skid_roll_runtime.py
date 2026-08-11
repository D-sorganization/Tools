"""Mutable runtime ledger isolated from the skid/roll state-machine policy."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

from ._vector_math import dot, norm, subtract
from .bounce_types import RepeatedBounceResult
from .contract_records import GroundSimulationRequest
from .contract_types import (
    GroundContactState,
    GroundEvent,
    GroundEventType,
    GroundPhase,
    GroundTrajectoryPoint,
)
from .impact_types import SphereProperties
from .skid_roll_dynamics import (
    advance_constant_motion,
    constant_motion_endpoint,
    constant_motion_energy_balance,
    contact_slip_velocity,
    holding_state,
    kinetic_energy,
    kinetic_energy_vectors,
    relative_path_distance,
    rolling_state,
    tangent,
)
from .surface_motion_types import (
    CancellationCheck,
    RigidMotion,
    SkidRollEnergyLedger,
    SkidRollResult,
    SkidRollSettings,
    SkidRollTermination,
    SkidRollTerminationReason,
    SurfaceKinematicSegment,
)
from .surface_resolver import SurfaceResolver

_CANONICAL_QUANTUM = 1e-11
_FLOATING_ERROR_MULTIPLIER = 64.0


@dataclass(frozen=True)
class AdvanceResult:
    """Report whether an exact finite-domain edge truncated one step."""

    boundary_crossed: bool


@dataclass
class SurfaceRun:
    """Accumulate one deterministic suffix without constructing wire output."""

    request: GroundSimulationRequest
    prefix: RepeatedBounceResult
    resolver: SurfaceResolver
    settings: SkidRollSettings
    is_cancelled: CancellationCheck
    body: SphereProperties
    state: GroundContactState
    phase: GroundPhase = GroundPhase.SKID
    trajectory: list[GroundTrajectoryPoint] = field(default_factory=list)
    events: list[GroundEvent] = field(default_factory=list)
    skid_distance_m: float = 0.0
    roll_distance_m: float = 0.0
    surface_work_j: float = 0.0
    physical_dissipation_j: float = 0.0
    canonical_energy_budget_j: float = 0.0
    step_count: int = 0
    next_grid_time_s: float = 0.0

    @property
    def first_contact_time_s(self) -> float:
        return float(self.prefix.events[0].time_s)

    @property
    def time_limit_s(self) -> float:
        return float(self.first_contact_time_s + self.request.max_time_s)

    def elapsed(self) -> float:
        return float(max(0.0, self.state.time_s - self.first_contact_time_s))

    def append_point(self, state: GroundContactState, phase: GroundPhase) -> None:
        """Append/replace a suffix point, never duplicating the prefix handoff."""
        handoff = self.prefix.handoff_state
        if handoff is None:
            raise RuntimeError("surface run requires a handoff")
        tolerance = self.settings.time_tolerance_s
        if state.time_s <= handoff.time_s + tolerance:
            return
        point = GroundTrajectoryPoint(
            state.time_s,
            state.frame,
            state.position_m,
            state.velocity_m_s,
            state.angular_velocity_rad_s,
            phase,
        )
        if (
            self.trajectory
            and abs(point.time_s - self.trajectory[-1].time_s) <= tolerance
        ):
            self.trajectory[-1] = point
        else:
            self.trajectory.append(point)

    def append_event(
        self,
        event_type: GroundEventType,
        before: GroundContactState,
        after: GroundContactState,
    ) -> bool:
        """Append one globally sequenced event when request capacity remains."""
        sequence = len(self.prefix.events) + len(self.events)
        if sequence >= self.request.max_events:
            return False
        self.events.append(
            GroundEvent(
                sequence,
                event_type,
                before.time_s,
                before.frame,
                before.position_m,
                before.velocity_m_s,
                after.velocity_m_s,
                before.angular_velocity_rad_s,
                after.angular_velocity_rad_s,
            )
        )
        return True

    def result(self, reason: SkidRollTerminationReason) -> SkidRollResult:
        """Build a validated internal suffix with passive energy accounting."""
        initial = self.prefix.handoff_state
        if initial is None:
            raise RuntimeError("surface run requires a handoff")
        before = kinetic_energy(initial, self.body)
        after = kinetic_energy(self.state, self.body)
        displacement = subtract(self.state.position_m, initial.position_m)
        gravity_work = self.body.mass_kg * dot(self.settings.gravity_m_s2, displacement)
        dissipation = before + gravity_work + self.surface_work_j - after
        energy_scale = max(
            1.0, before, after, abs(gravity_work), abs(self.surface_work_j)
        )
        canonical_tolerance = self.canonical_energy_budget_j + _floating_tolerance(
            energy_scale, self.step_count + 1
        )
        if dissipation < -canonical_tolerance:
            raise ValueError("surface run violates passive energy accounting")
        energy = SkidRollEnergyLedger(
            before,
            after,
            gravity_work,
            self.surface_work_j,
            max(0.0, dissipation),
        )
        return SkidRollResult(
            self.request.request_id,
            self.request.surface.surface_id,
            self.request.surface.frame,
            self.settings.model_id,
            self.settings.model_version,
            self.prefix.request_fingerprint_sha256,
            tuple(self.trajectory),
            tuple(self.events),
            self.state,
            self.skid_distance_m,
            self.roll_distance_m,
            energy,
            SkidRollTermination(reason, self.state.time_s, self.elapsed()),
        )

    def emit_grid_points(
        self,
        start: GroundContactState,
        motion: RigidMotion,
        duration_s: float,
    ) -> None:
        """Emit global-grid samples strictly after the #4270 handoff."""
        terminal = start.time_s + duration_s
        tolerance = self.settings.time_tolerance_s
        while self.next_grid_time_s <= start.time_s + tolerance:
            self.next_grid_time_s += self.request.output_interval_s
        while self.next_grid_time_s < terminal - tolerance:
            sample = advance_constant_motion(
                start,
                motion,
                self.next_grid_time_s - start.time_s,
            )
            self.append_point(sample, self.phase)
            self.next_grid_time_s += self.request.output_interval_s

    def advance(self, motion: RigidMotion, duration_s: float) -> AdvanceResult:
        """Advance to an event/step endpoint and account work and path."""
        segment = SurfaceKinematicSegment(
            self.state.position_m,
            self.state.velocity_m_s,
            motion.acceleration_m_s2,
            duration_s,
        )
        crossing = self.resolver.first_crossing(segment)
        actual_duration = crossing.time_offset_s if crossing is not None else duration_s
        start = self.state
        self.emit_grid_points(start, motion, actual_duration)
        self.state = advance_constant_motion(start, motion, actual_duration)
        self._account_step(start, motion, actual_duration)
        return AdvanceResult(crossing is not None)

    def _account_step(
        self,
        start: GroundContactState,
        motion: RigidMotion,
        duration_s: float,
    ) -> None:
        surface = self.request.surface
        relative = tangent(
            subtract(start.velocity_m_s, surface.surface_velocity_m_s),
            surface.normal_unit,
        )
        distance = relative_path_distance(
            relative, motion.acceleration_m_s2, duration_s
        )
        if self.phase is GroundPhase.SKID:
            self.skid_distance_m += distance
        else:
            self.roll_distance_m += distance
        surface_work = (
            dot(motion.contact_force_n, surface.surface_velocity_m_s) * duration_s
        )
        self.surface_work_j += surface_work
        segment_dissipation = constant_motion_energy_balance(
            start,
            motion,
            duration_s,
            self.body,
            self.settings.gravity_m_s2,
            surface.surface_velocity_m_s,
        )
        position, velocity, spin = constant_motion_endpoint(start, motion, duration_s)
        raw_energy = kinetic_energy_vectors(velocity, spin, self.body)
        gravity_work = self.body.mass_kg * dot(
            self.settings.gravity_m_s2, subtract(position, start.position_m)
        )
        segment_scale = max(
            1.0,
            kinetic_energy(start, self.body),
            raw_energy,
            abs(gravity_work),
            abs(surface_work),
        )
        if segment_dissipation < -_floating_tolerance(segment_scale):
            raise ValueError("surface run violates passive energy accounting")
        self.physical_dissipation_j += max(0.0, segment_dissipation)
        self.canonical_energy_budget_j += _canonical_snap_energy_budget(
            (position, velocity, spin),
            self.state,
            self.body,
            self.settings.gravity_m_s2,
        )

    def rolling_projection(self) -> GroundContactState:
        """Return a bounded no-slip projection and account numerical energy."""
        before = self.state
        return self._bounded_projection(
            before,
            rolling_state(before, self.request.surface, self.body),
            correction_tolerance_m_s=self.settings.slip_tolerance_m_s,
        )

    def holding_projection(self) -> GroundContactState:
        """Return an exactly co-moving state inside the same projection bounds."""
        before = self.state
        return self._bounded_projection(
            before,
            holding_state(before, self.request.surface),
            correction_tolerance_m_s=self.settings.velocity_tolerance_m_s,
        )

    def _bounded_projection(
        self,
        before: GroundContactState,
        after: GroundContactState,
        *,
        correction_tolerance_m_s: float,
    ) -> GroundContactState:
        slip = norm(contact_slip_velocity(before, self.request.surface, self.body))
        tolerance = self.settings.slip_tolerance_m_s
        if slip > tolerance + _floating_tolerance(max(1.0, slip)):
            raise ValueError("surface run violates passive energy accounting")
        velocity_change = norm(subtract(after.velocity_m_s, before.velocity_m_s))
        spin_change = norm(
            subtract(
                after.angular_velocity_rad_s,
                before.angular_velocity_rad_s,
            )
        )
        component_rounding = math.sqrt(3.0) * _CANONICAL_QUANTUM
        if velocity_change > correction_tolerance_m_s + component_rounding:
            raise ValueError("surface run violates passive energy accounting")
        if (
            spin_change
            > correction_tolerance_m_s / self.body.radius_m + component_rounding
        ):
            raise ValueError("surface run violates passive energy accounting")
        energy_creation = kinetic_energy(after, self.body) - kinetic_energy(
            before, self.body
        )
        self.canonical_energy_budget_j += max(0.0, energy_creation)
        return after


def _floating_tolerance(scale: float, operations: int = 1) -> float:
    return float(
        _FLOATING_ERROR_MULTIPLIER
        * sys.float_info.epsilon
        * max(1.0, scale)
        * operations
    )


def _component_error_bound(value: float) -> float:
    return float(
        0.5 * _CANONICAL_QUANTUM + 4.0 * sys.float_info.epsilon * max(1.0, abs(value))
    )


def _canonical_snap_energy_budget(
    raw: tuple[tuple[float, float, float], ...],
    canonical: GroundContactState,
    body: SphereProperties,
    gravity_m_s2: tuple[float, float, float],
) -> float:
    canonical_vectors = (
        canonical.position_m,
        canonical.velocity_m_s,
        canonical.angular_velocity_rad_s,
    )
    error_norms: list[float] = []
    for raw_vector, canonical_vector in zip(raw, canonical_vectors, strict=True):
        bounds = tuple(_component_error_bound(value) for value in raw_vector)
        for actual, expected, bound in zip(
            canonical_vector, raw_vector, bounds, strict=True
        ):
            if abs(actual - expected) > bound:
                raise ValueError("surface run exceeds canonical quantization bound")
        error_norms.append(math.sqrt(sum(bound * bound for bound in bounds)))
    position_error, velocity_error, spin_error = error_norms
    return float(
        body.mass_kg * norm(gravity_m_s2) * position_error
        + body.mass_kg * (norm(raw[1]) * velocity_error + 0.5 * velocity_error**2)
        + body.inertia_kg_m2 * (norm(raw[2]) * spin_error + 0.5 * spin_error**2)
    )


__all__ = ["AdvanceResult", "SurfaceRun"]
