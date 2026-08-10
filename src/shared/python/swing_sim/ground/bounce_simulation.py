"""Deterministic bounded state machine for repeated sphere-plane impacts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .bounce_kinematics import (
    ballistic_state,
    contact_state_after_hop,
    interpolate_first_contact,
    trajectory_point,
)
from .bounce_types import (
    BounceAirSegment,
    BounceModelSettings,
    BounceTermination,
    BounceTerminationReason,
    CancellationCheck,
    RepeatedBounceResult,
)
from .contract_records import GroundSimulationRequest
from .contract_types import (
    GroundContactState,
    GroundEvent,
    GroundEventType,
    GroundPhase,
    GroundTrajectoryPoint,
)
from .impact_impulse import resolve_sphere_plane_impact
from .impact_types import ImpactImpulseResult, SphereProperties

_MATERIAL_LIMITATION = (
    "Rigid restitution v1 does not use firmness_pa, hardness_fraction, "
    "grass_height_m, compressibility_fraction, compression_damping_fraction, "
    "turf_density_kg_m3, moisture_fraction, or rolling_resistance."
)
_HANDOFF_LIMITATION = (
    "The prefix ends at settled-to-skid; #4271 owns skid/roll, total distance, "
    "and the final GroundSimulationResult."
)
_EVENT_FIRST_CONTACT = GroundEventType("first_contact")
_EVENT_BOUNCE = GroundEventType("bounce")
_PHASE_IMPACT = GroundPhase("impact")
_PHASE_BOUNCE = GroundPhase("bounce")
_PHASE_SKID = GroundPhase("skid")
_SETTLED_TO_SKID = BounceTerminationReason("settled_to_skid")
_CANCELLED = BounceTerminationReason("cancelled")
_TIME_LIMIT = BounceTerminationReason("time_limit")
_EVENT_LIMIT = BounceTerminationReason("event_limit")
_NO_RECONTACT = BounceTerminationReason("no_recontact")
_NUMERICAL_FAILURE = BounceTerminationReason("numerical_failure")


@dataclass
class _BounceRun:
    request: GroundSimulationRequest
    settings: BounceModelSettings
    is_cancelled: CancellationCheck
    points: list[GroundTrajectoryPoint] = field(default_factory=list)
    events: list[GroundEvent] = field(default_factory=list)
    impacts: list[ImpactImpulseResult] = field(default_factory=list)
    airborne_segments: list[BounceAirSegment] = field(default_factory=list)
    first_contact_time_s: float | None = None
    next_grid_time_s: float | None = None

    def elapsed(self, time_s: float) -> float:
        """Return elapsed ground time from first physical contact."""
        if self.first_contact_time_s is None:
            return 0.0
        return max(0.0, time_s - self.first_contact_time_s)

    def result(
        self,
        reason: BounceTerminationReason,
        time_s: float,
        handoff: GroundContactState | None = None,
    ) -> RepeatedBounceResult:
        """Build one validated prefix result from accumulated state."""
        return RepeatedBounceResult(
            request_id=self.request.request_id,
            surface_id=self.request.surface.surface_id,
            frame=self.request.surface.frame,
            model_id=self.settings.model_id,
            model_version=self.settings.model_version,
            trajectory=tuple(self.points),
            events=tuple(self.events),
            impacts=tuple(self.impacts),
            airborne_segments=tuple(self.airborne_segments),
            handoff_state=handoff,
            termination=BounceTermination(reason, time_s, self.elapsed(time_s)),
            warnings=(_MATERIAL_LIMITATION, _HANDOFF_LIMITATION),
        )

    def append_point(self, point: GroundTrajectoryPoint) -> None:
        """Append or replace one time-coincident point deterministically."""
        if self.points:
            delta = point.time_s - self.points[-1].time_s
            if abs(delta) <= self.settings.time_tolerance_s:
                self.points[-1] = point
                return
            if delta < 0.0:
                raise ValueError("bounce trajectory cannot move backward in time")
        self.points.append(point)

    def add_impact(
        self,
        incoming: GroundContactState,
        event_type: GroundEventType,
    ) -> ImpactImpulseResult:
        """Resolve and record one impact using an optional capture override."""
        incoming_speed = -self.request.surface.relative_normal_speed_m_s(incoming)
        restitution = 0.0 if incoming_speed <= self.settings.capture_speed_m_s else None
        impact = resolve_sphere_plane_impact(
            incoming,
            self.request.surface,
            SphereProperties(
                self.request.ball_radius_m,
                self.request.ball_mass_kg,
                self.request.rotational_inertia_factor,
            ),
            normal_restitution=restitution,
        )
        event = GroundEvent(
            sequence=len(self.events),
            event_type=event_type,
            time_s=incoming.time_s,
            frame=incoming.frame,
            position_m=incoming.position_m,
            velocity_before_m_s=incoming.velocity_m_s,
            velocity_after_m_s=impact.state_after.velocity_m_s,
            angular_velocity_before_rad_s=incoming.angular_velocity_rad_s,
            angular_velocity_after_rad_s=impact.state_after.angular_velocity_rad_s,
        )
        phase = _PHASE_IMPACT if event_type is _EVENT_FIRST_CONTACT else _PHASE_BOUNCE
        self.append_point(trajectory_point(impact.state_after, phase))
        self.events.append(event)
        self.impacts.append(impact)
        return impact

    def add_airborne_samples(
        self,
        outgoing: GroundContactState,
        terminal_time_s: float,
        *,
        include_terminal: bool,
    ) -> None:
        """Add global-grid hop samples, optionally including the terminal state."""
        if self.next_grid_time_s is None:
            raise RuntimeError("airborne sampling requires an initialized output grid")
        tolerance = self.settings.time_tolerance_s
        next_grid_time_s = self.next_grid_time_s
        while next_grid_time_s <= outgoing.time_s + tolerance:
            next_grid_time_s += self.request.output_interval_s
        while next_grid_time_s < terminal_time_s - tolerance:
            elapsed = next_grid_time_s - outgoing.time_s
            state = ballistic_state(outgoing, elapsed, self.settings)
            self.append_point(trajectory_point(state, _PHASE_BOUNCE))
            next_grid_time_s += self.request.output_interval_s
        self.next_grid_time_s = next_grid_time_s
        if include_terminal:
            elapsed = terminal_time_s - outgoing.time_s
            terminal = ballistic_state(outgoing, elapsed, self.settings)
            self.append_point(trajectory_point(terminal, _PHASE_BOUNCE))
        else:
            terminal = ballistic_state(
                outgoing,
                terminal_time_s - outgoing.time_s,
                self.settings,
            )
        self.airborne_segments.append(
            BounceAirSegment(
                start_time_s=outgoing.time_s,
                end_time_s=terminal_time_s,
                start_position_m=outgoing.position_m,
                end_position_m=terminal.position_m,
                horizontal_distance_m=(
                    (terminal.position_m[0] - outgoing.position_m[0]) ** 2
                    + (terminal.position_m[2] - outgoing.position_m[2]) ** 2
                )
                ** 0.5,
                completed_at_contact=not include_terminal,
            )
        )

    def run_after_impact(self, impact: ImpactImpulseResult) -> RepeatedBounceResult:
        """Advance hops until a bounded state-machine termination occurs."""
        current = impact
        while True:
            outgoing = current.state_after
            if current.effective_restitution == 0.0:
                self.append_point(trajectory_point(outgoing, _PHASE_SKID))
                return self.result(_SETTLED_TO_SKID, outgoing.time_s, outgoing)
            if self.is_cancelled():
                return self.result(_CANCELLED, outgoing.time_s)
            if len(self.events) >= self.request.max_events:
                return self.result(_EVENT_LIMIT, outgoing.time_s)
            incoming = contact_state_after_hop(outgoing, self.request, self.settings)
            if incoming is None:
                return self.result(_NO_RECONTACT, outgoing.time_s)
            if self.first_contact_time_s is None:
                raise RuntimeError("bounce propagation requires first-contact time")
            time_limit = self.first_contact_time_s + self.request.max_time_s
            if incoming.time_s > time_limit + self.settings.time_tolerance_s:
                self.add_airborne_samples(outgoing, time_limit, include_terminal=True)
                return self.result(_TIME_LIMIT, time_limit)
            self.add_airborne_samples(outgoing, incoming.time_s, include_terminal=False)
            current = self.add_impact(incoming, _EVENT_BOUNCE)


def _never_cancelled() -> bool:
    return False


def simulate_repeated_bounce(
    request: GroundSimulationRequest,
    settings: BounceModelSettings | None = None,
    *,
    is_cancelled: CancellationCheck | None = None,
) -> RepeatedBounceResult:
    """Run the deterministic impact/bounce prefix for a strict v1 request.

    The returned prefix intentionally excludes skid, roll, rest, total distance,
    terrain deformation, and any fabricated final ``GroundSimulationResult``.
    Invalid request/configuration records raise before state-machine execution.
    """
    if type(request) is not GroundSimulationRequest:
        raise ValueError("bounce simulation requires an exact ground request")
    resolved_settings = BounceModelSettings() if settings is None else settings
    if type(resolved_settings) is not BounceModelSettings:
        raise ValueError("settings must be an exact BounceModelSettings record")
    cancellation = _never_cancelled if is_cancelled is None else is_cancelled
    if not callable(cancellation):
        raise ValueError("is_cancelled must be callable")
    run = _BounceRun(request, resolved_settings, cancellation)
    if cancellation():
        return run.result(
            _CANCELLED,
            request.last_separated_state.time_s,
        )
    contact = interpolate_first_contact(request)
    run.first_contact_time_s = contact.time_s
    run.next_grid_time_s = contact.time_s + request.output_interval_s
    try:
        impact = run.add_impact(contact, _EVENT_FIRST_CONTACT)
        return run.run_after_impact(impact)
    except (ArithmeticError, ValueError):
        terminal_time = run.points[-1].time_s if run.points else contact.time_s
        return run.result(_NUMERICAL_FAILURE, terminal_time)


__all__ = ["simulate_repeated_bounce"]
