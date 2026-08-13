"""Mutable runtime ledger isolated from the skid/roll state-machine policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from ._vector_math import dot, subtract
from .bounce_types import RepeatedBounceResult
from .contract_records import GroundSimulationRequest
from .contract_types import (
    GroundContactState,
    GroundEvent,
    GroundEventType,
    GroundPhase,
    GroundSurfaceProfile,
    GroundTrajectoryPoint,
)
from .impact_types import SphereProperties
from .regional_surface_types import (
    SurfaceRegionTransition,
    SurfaceRegionTransitionCrossing,
)
from .skid_roll_dynamics import (
    advance_constant_motion,
    kinetic_energy,
    relative_path_distance,
    tangent,
)
from .skid_roll_result_types import SkidRollResult
from .surface_motion_types import (
    CancellationCheck,
    RigidMotion,
    SkidRollEnergyLedger,
    SkidRollSettings,
    SkidRollTermination,
    SkidRollTerminationReason,
    SurfaceKinematicSegment,
)
from .surface_resolver import SurfaceResolver


@dataclass(frozen=True)
class AdvanceResult:
    """Report the exact domain or regional boundary truncating one step."""

    boundary_crossed: bool
    transition: SurfaceRegionTransitionCrossing | None = None


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
    active_surface: GroundSurfaceProfile
    active_region_id: str | None = None
    phase: GroundPhase = GroundPhase.SKID
    trajectory: list[GroundTrajectoryPoint] = field(default_factory=list)
    events: list[GroundEvent] = field(default_factory=list)
    surface_transitions: list[SurfaceRegionTransition] = field(default_factory=list)
    skid_distance_m: float = 0.0
    roll_distance_m: float = 0.0
    gravity_work_j: float = 0.0
    surface_work_j: float = 0.0
    step_count: int = 0
    surface_transition_count: int = 0
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
        dissipation = before + self.gravity_work_j + self.surface_work_j - after
        tolerance = 1e-9 + 1e-9 * max(before, after, abs(self.surface_work_j))
        if dissipation < -tolerance:
            raise ValueError("surface run violates passive energy accounting")
        energy = SkidRollEnergyLedger(
            before,
            after,
            self.gravity_work_j,
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
            tuple(self.surface_transitions),
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
        transition = self.resolver.first_transition(segment, self.active_region_id)
        outer_is_first = crossing is not None and (
            transition is None
            or crossing.time_offset_s
            <= transition.time_offset_s + self.settings.time_tolerance_s
        )
        if outer_is_first and crossing is not None:
            actual_duration = crossing.time_offset_s
            selected_transition = None
        elif transition is not None:
            actual_duration = transition.time_offset_s
            selected_transition = transition
        else:
            actual_duration = duration_s
            selected_transition = None
        start = self.state
        self.emit_grid_points(start, motion, actual_duration)
        self.state = advance_constant_motion(start, motion, actual_duration)
        self._account_step(start, motion, actual_duration)
        return AdvanceResult(outer_is_first, selected_transition)

    def _account_step(
        self,
        start: GroundContactState,
        motion: RigidMotion,
        duration_s: float,
    ) -> None:
        surface = self.active_surface
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
        displacement = subtract(self.state.position_m, start.position_m)
        self.gravity_work_j += self.body.mass_kg * dot(
            self.settings.gravity_m_s2,
            displacement,
        )
        self.surface_work_j += (
            dot(
                motion.contact_force_n,
                surface.surface_velocity_m_s,
            )
            * duration_s
        )


__all__ = ["AdvanceResult", "SurfaceRun"]
