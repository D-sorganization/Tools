"""Deterministic bounded skid-to-roll solver for coplanar material regions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ._vector_math import cross, norm, subtract
from .bounce_types import RepeatedBounceResult
from .contract_records import GroundSimulationRequest
from .contract_types import (
    GroundContactState,
    GroundEventType,
    GroundPhase,
    Vector3,
)
from .impact_types import SphereProperties
from .regional_surface_types import (
    SurfaceRegionTransition,
    SurfaceRegionTransitionCrossing,
)
from .skid_roll_dynamics import (
    bounded_closing_duration,
    contact_slip_velocity,
    holding_kinematics,
    rolling_kinematics,
    skid_kinematics,
    stable_at_zero_speed,
    static_rolling_feasible,
    tangent,
    time_to_vector_zero,
)
from .skid_roll_result_types import SkidRollResult
from .skid_roll_runtime import SurfaceRun
from .skid_roll_validation import validate_surface_run_inputs
from .surface_motion_types import (
    CancellationCheck,
    PlanarSurfaceDomain,
    SkidRollSettings,
    SkidRollTerminationReason,
)
from .surface_resolver import SurfaceResolver

_ZERO: Vector3 = (0.0, 0.0, 0.0)
_PHASE_SKID = GroundPhase("skid")
_PHASE_ROLL = GroundPhase("roll")
_PHASE_REST = GroundPhase("rest")


@dataclass(frozen=True)
class SkidRollExecution:
    """Optional numerical, domain, and cancellation controls for one run."""

    settings: SkidRollSettings = field(default_factory=SkidRollSettings)
    resolver: SurfaceResolver | None = None
    is_cancelled: CancellationCheck | None = None

    def __post_init__(self) -> None:
        if type(self.settings) is not SkidRollSettings:
            raise ValueError("execution settings must be exact")
        if self.resolver is not None and type(self.resolver) is not SurfaceResolver:
            raise ValueError("execution resolver must be exact")
        if self.is_cancelled is not None and not callable(self.is_cancelled):
            raise ValueError("execution cancellation hook must be callable")


def _never_cancelled() -> bool:
    return False


def _typed_result(
    run: SurfaceRun,
    reason: SkidRollTerminationReason,
) -> SkidRollResult:
    """Return the runtime result across the skipped-import type boundary."""
    return run.result(reason)


def _event_result(
    run: SurfaceRun,
    event_type: GroundEventType,
    after: GroundContactState,
    reason: SkidRollTerminationReason | None = None,
) -> SkidRollResult | None:
    phase = {
        GroundEventType.SKID_TO_ROLL: _PHASE_ROLL,
        GroundEventType.REST: _PHASE_REST,
    }.get(event_type, run.phase)
    before = run.state
    if not run.append_event(event_type, before, after):
        run.append_point(before, run.phase)
        return _typed_result(run, SkidRollTerminationReason.EVENT_LIMIT)
    run.state = after
    run.phase = phase
    run.append_point(after, phase)
    return None if reason is None else _typed_result(run, reason)


def _can_rest(run: SurfaceRun) -> bool:
    surface = run.active_surface
    return (
        surface.surface_velocity_m_s == _ZERO
        and stable_at_zero_speed(surface, run.body, run.settings.gravity_m_s2)
        and norm(run.state.velocity_m_s) <= run.settings.velocity_tolerance_m_s
        and norm(run.state.angular_velocity_rad_s)
        <= run.settings.angular_tolerance_rad_s
    )


def _rest_or_hold(run: SurfaceRun) -> SkidRollResult | None:
    if _can_rest(run):
        stopped = replace(run.state, velocity_m_s=_ZERO, angular_velocity_rad_s=_ZERO)
        return _event_result(
            run,
            GroundEventType.REST,
            stopped,
            SkidRollTerminationReason.REST,
        )
    return None


def _transition_to_roll(run: SurfaceRun) -> SkidRollResult | None:
    if not static_rolling_feasible(
        run.active_surface, run.body, run.settings.gravity_m_s2
    ):
        return _typed_result(run, SkidRollTerminationReason.UNSUPPORTED_SURFACE)
    rolled = run.rolling_projection()
    return _event_result(run, GroundEventType.SKID_TO_ROLL, rolled)


def _surface_transition(
    run: SurfaceRun,
    transition: SurfaceRegionTransitionCrossing,
) -> SkidRollResult | None:
    """Record one continuous material change or return a typed bound."""
    if type(transition) is not SurfaceRegionTransitionCrossing:
        raise RuntimeError("surface transition outcome must be exact")
    if run.surface_transition_count >= run.settings.max_surface_transitions:
        run.append_point(run.state, run.phase)
        return run.result(SkidRollTerminationReason.SURFACE_TRANSITION_LIMIT)
    before = run.state
    sequence = len(run.prefix.events) + len(run.events)
    from_surface_id = run.active_surface.surface_id
    if not run.append_event(GroundEventType.SURFACE_TRANSITION, before, before):
        run.append_point(before, run.phase)
        return run.result(SkidRollTerminationReason.EVENT_LIMIT)
    run.surface_transitions.append(
        SurfaceRegionTransition(
            sequence,
            before.time_s,
            before.position_m,
            transition.from_region_id,
            transition.to_region_id,
            from_surface_id,
            transition.to_surface.surface_id,
        )
    )
    run.active_surface = transition.to_surface
    run.active_region_id = transition.to_region_id
    run.surface_transition_count += 1
    run.append_point(before, run.phase)
    return None


def _skid_step(run: SurfaceRun, duration_s: float) -> SkidRollResult | None:
    slip = contact_slip_velocity(run.state, run.active_surface, run.body)
    if norm(slip) <= run.settings.slip_tolerance_m_s:
        return _transition_to_roll(run)
    motion = skid_kinematics(
        run.state,
        run.active_surface,
        run.body,
        run.settings.gravity_m_s2,
    )
    transition = time_to_vector_zero(
        slip,
        motion.contact_slip_acceleration_m_s2,
        tolerance=run.settings.slip_tolerance_m_s,
    )
    reaches_roll = transition is not None and transition <= duration_s
    advance_for = (
        transition
        if reaches_roll and transition is not None
        else bounded_closing_duration(
            slip,
            motion.contact_slip_acceleration_m_s2,
            duration_s,
        )
    )
    outcome = run.advance(motion, advance_for)
    if outcome.boundary_crossed:
        return _left_surface(run)
    if outcome.transition is not None:
        return _surface_transition(run, outcome.transition)
    if reaches_roll:
        return _transition_to_roll(run)
    return None


def _roll_step(run: SurfaceRun, duration_s: float) -> SkidRollResult | None:
    surface = run.active_surface
    if not static_rolling_feasible(surface, run.body, run.settings.gravity_m_s2):
        return _typed_result(run, SkidRollTerminationReason.UNSUPPORTED_SURFACE)
    run.state = run.rolling_projection()
    relative = tangent(
        subtract(run.state.velocity_m_s, surface.surface_velocity_m_s),
        surface.normal_unit,
    )
    if norm(relative) <= run.settings.velocity_tolerance_m_s:
        resting = _rest_or_hold(run)
        if resting is not None:
            return resting
        if stable_at_zero_speed(surface, run.body, run.settings.gravity_m_s2):
            run.state = run.holding_projection()
            motion = holding_kinematics(
                surface,
                run.body,
                run.settings.gravity_m_s2,
            )
            if _can_rest(run):
                handoff = run.prefix.handoff_state
                if handoff is None:
                    raise RuntimeError("surface run requires a handoff")
                if run.state.time_s <= handoff.time_s + run.settings.time_tolerance_s:
                    outcome = run.advance(motion, duration_s)
                    if outcome.boundary_crossed:
                        return _left_surface(run)
                return _rest_or_hold(run)
            outcome = run.advance(
                motion,
                duration_s,
            )
            return _left_surface(run) if outcome.boundary_crossed else None
    motion = rolling_kinematics(run.state, surface, run.body, run.settings.gravity_m_s2)
    stop = time_to_vector_zero(
        relative,
        motion.acceleration_m_s2,
        tolerance=run.settings.velocity_tolerance_m_s,
    )
    reaches_zero = stop is not None and stop <= duration_s
    advance_for = (
        stop
        if reaches_zero and stop is not None
        else (
            bounded_closing_duration(
                relative,
                motion.acceleration_m_s2,
                duration_s,
            )
            if norm(cross(relative, motion.acceleration_m_s2)) > 1e-12
            else duration_s
        )
    )
    outcome = run.advance(motion, advance_for)
    if outcome.boundary_crossed:
        return _left_surface(run)
    if outcome.transition is not None:
        return _surface_transition(run, outcome.transition)
    if reaches_zero:
        run.state = run.rolling_projection()
        return _rest_or_hold(run)
    return None


def _left_surface(run: SurfaceRun) -> SkidRollResult:
    result = _event_result(
        run,
        GroundEventType.LEFT_SURFACE,
        run.state,
        SkidRollTerminationReason.LEFT_SURFACE,
    )
    if result is None:
        raise RuntimeError("terminal left-surface event must produce a result")
    return result


def _run_loop(run: SurfaceRun) -> SkidRollResult:
    while True:
        if run.is_cancelled():
            return _typed_result(run, SkidRollTerminationReason.CANCELLED)
        if run.step_count >= run.settings.max_steps:
            return _typed_result(run, SkidRollTerminationReason.STEP_LIMIT)
        remaining = run.time_limit_s - run.state.time_s
        if remaining <= run.settings.time_tolerance_s:
            run.append_point(run.state, run.phase)
            return _typed_result(run, SkidRollTerminationReason.TIME_LIMIT)
        duration = min(run.settings.integration_step_s, remaining)
        run.step_count += 1
        result = (
            _skid_step(run, duration)
            if run.phase is _PHASE_SKID
            else _roll_step(run, duration)
        )
        if result is not None:
            return result


def simulate_skid_roll(
    request: GroundSimulationRequest,
    prefix: RepeatedBounceResult,
    execution: SkidRollExecution | None = None,
) -> SkidRollResult:
    """Propagate a settled #4270 handoff through skid, roll, and bounded stop."""
    selected_execution = SkidRollExecution() if execution is None else execution
    if type(selected_execution) is not SkidRollExecution:
        raise ValueError("execution must be an exact SkidRollExecution record")
    selected_settings = selected_execution.settings
    selected_resolver = (
        SurfaceResolver(PlanarSurfaceDomain.unbounded(request.surface))
        if selected_execution.resolver is None
        else selected_execution.resolver
    )
    cancellation = (
        _never_cancelled
        if selected_execution.is_cancelled is None
        else selected_execution.is_cancelled
    )
    handoff = validate_surface_run_inputs(
        request,
        prefix,
        selected_resolver,
        selected_settings,
    )
    body = SphereProperties(
        request.ball_radius_m,
        request.ball_mass_kg,
        request.rotational_inertia_factor,
    )
    next_grid = prefix.events[0].time_s + request.output_interval_s
    run = SurfaceRun(
        request,
        prefix,
        selected_resolver,
        selected_settings,
        cancellation,
        body,
        handoff,
        request.surface,
        next_grid_time_s=next_grid,
    )
    return _run_loop(run)


__all__ = ["SkidRollExecution", "simulate_skid_roll"]
