//! Bounded skid-to-roll propagation on one immutable unbounded plane.

use super::bounce_runtime::BouncePrefix;
use super::execution_v1::{
    GroundReferencePhaseV1, GroundReferenceRuntimeCodeV1, GroundReferenceRuntimeErrorV1,
    SkidRollExecutionSettingsV1,
};
use super::request_v1::FlightToGroundRequestV1;
use super::resource_limits::ensure_trajectory_capacity;
use super::result_v1::{
    GroundEventTypeV1, GroundEventV1, GroundPhaseV1, GroundTerminationReasonV1,
    GroundTrajectoryPointV1,
};
use super::runtime_math::{
    advance, closing_duration, norm, time_to_zero, Body, Motion, OutputSchedule, State,
    TimelineAppend, TimelineError, WireTimeline,
};
use super::surface_dynamics::{
    can_rest, center_relative_velocity, contact_slip, rolling_feasible, rolling_motion,
    rolling_state, skid_motion, LedgerStep, Phase, SurfaceLedger,
};
use super::surface_events::SurfaceEventContext;

const ZERO: [f64; 3] = [0.0; 3];

pub(super) struct SurfaceSuffix {
    pub trajectory: Vec<GroundTrajectoryPointV1>,
    pub events: Vec<GroundEventV1>,
    pub state: State,
    pub skid_distance: f64,
    pub roll_distance: f64,
    pub termination: GroundTerminationReasonV1,
}

struct SurfaceRun<'a> {
    request: &'a FlightToGroundRequestV1,
    settings: &'a SkidRollExecutionSettingsV1,
    fingerprint: &'a str,
    body: Body,
    initial_state: State,
    state: State,
    phase: Phase,
    prefix_event_count: usize,
    prefix_trajectory_count: usize,
    trajectory: Vec<GroundTrajectoryPointV1>,
    events: Vec<GroundEventV1>,
    ledger: SurfaceLedger,
    step_count: u64,
    output_schedule: OutputSchedule,
    time_origin: f64,
    handoff_time: f64,
    timeline: WireTimeline,
}

pub(super) fn simulate_surface<C>(
    request: &FlightToGroundRequestV1,
    settings: &SkidRollExecutionSettingsV1,
    prefix: &BouncePrefix,
    fingerprint: &str,
    is_cancelled: &mut C,
) -> Result<SurfaceSuffix, GroundReferenceRuntimeErrorV1>
where
    C: FnMut() -> bool,
{
    let mut run = SurfaceRun::new(request, settings, prefix, fingerprint)?;
    loop {
        if is_cancelled() {
            return Err(run.error(GroundReferenceRuntimeCodeV1::Cancelled, "cancelled"));
        }
        if run.step_count >= settings.max_steps {
            return Err(run.error(GroundReferenceRuntimeCodeV1::ExecutionFailure, "step_limit"));
        }
        let remaining = run.request.max_time_s - run.state.time;
        if remaining <= settings.time_tolerance_s {
            run.append_point(run.state, run.phase.wire())?;
            return run.finish(GroundTerminationReasonV1::TimeLimit);
        }
        let duration = settings.integration_step_s.min(remaining);
        run.step_count += 1;
        if let Some(reason) = run.step(duration, is_cancelled)? {
            return run.finish(reason);
        }
    }
}

impl<'a> SurfaceRun<'a> {
    fn new(
        request: &'a FlightToGroundRequestV1,
        settings: &'a SkidRollExecutionSettingsV1,
        prefix: &BouncePrefix,
        fingerprint: &'a str,
    ) -> Result<Self, GroundReferenceRuntimeErrorV1> {
        let last_wire_time = prefix
            .trajectory
            .last()
            .map(|point| point.time_s)
            .ok_or_else(|| {
                GroundReferenceRuntimeErrorV1::new(
                    GroundReferenceRuntimeCodeV1::NumericalFailure,
                    GroundReferencePhaseV1::SkidRoll,
                    "trajectory_order",
                    fingerprint.to_owned(),
                )
            })?;
        Ok(Self {
            request,
            settings,
            fingerprint,
            body: Body::from_request(request),
            initial_state: prefix.handoff,
            state: prefix.handoff,
            phase: Phase::Skid,
            prefix_event_count: prefix.events.len(),
            prefix_trajectory_count: prefix.trajectory.len(),
            trajectory: Vec::new(),
            events: Vec::new(),
            ledger: SurfaceLedger::default(),
            step_count: 0,
            output_schedule: prefix.output_schedule,
            time_origin: prefix.time_origin,
            handoff_time: prefix.handoff.time,
            timeline: WireTimeline::anchored(prefix.handoff.time, last_wire_time),
        })
    }

    fn step(
        &mut self,
        duration: f64,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Option<GroundTerminationReasonV1>, GroundReferenceRuntimeErrorV1> {
        match self.phase {
            Phase::Skid => self.skid_step(duration, is_cancelled),
            Phase::Roll => self.roll_step(duration, is_cancelled),
        }
    }

    fn skid_step(
        &mut self,
        duration: f64,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Option<GroundTerminationReasonV1>, GroundReferenceRuntimeErrorV1> {
        let slip = contact_slip(self.state, &self.request.surface, self.body);
        if norm(slip) <= self.settings.slip_tolerance_m_s {
            return self.transition_to_roll();
        }
        let motion = skid_motion(
            self.state,
            &self.request.surface,
            self.body,
            self.settings.gravity_m_s2,
        )
        .ok_or_else(|| self.numerical("skid_direction"))?;
        let transition = time_to_zero(
            slip,
            motion.slip_acceleration,
            self.settings.slip_tolerance_m_s,
        );
        let transition = transition.filter(|time| *time <= duration);
        let reaches_roll = transition.is_some();
        let advance_for = transition
            .unwrap_or_else(|| closing_duration(slip, motion.slip_acceleration, duration));
        self.advance(motion, advance_for, is_cancelled)?;
        if reaches_roll {
            return self.transition_to_roll();
        }
        Ok(None)
    }

    fn transition_to_roll(
        &mut self,
    ) -> Result<Option<GroundTerminationReasonV1>, GroundReferenceRuntimeErrorV1> {
        if !rolling_feasible(&self.request.surface, self.body, self.settings.gravity_m_s2) {
            return Err(self.error(
                GroundReferenceRuntimeCodeV1::ExecutionFailure,
                "unsupported_surface",
            ));
        }
        let after = rolling_state(self.state, &self.request.surface, self.body)
            .map_err(|error| self.numerical(error.reason()))?;
        if !self.append_event(GroundEventTypeV1::SkidToRoll, self.state, after)? {
            self.append_point(self.state, self.phase.wire())?;
            return Ok(Some(GroundTerminationReasonV1::EventLimit));
        }
        self.state = after;
        self.phase = Phase::Roll;
        self.append_point(after, GroundPhaseV1::Roll)?;
        Ok(None)
    }

    fn roll_step(
        &mut self,
        duration: f64,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Option<GroundTerminationReasonV1>, GroundReferenceRuntimeErrorV1> {
        if !rolling_feasible(&self.request.surface, self.body, self.settings.gravity_m_s2) {
            return Err(self.error(
                GroundReferenceRuntimeCodeV1::ExecutionFailure,
                "unsupported_surface",
            ));
        }
        self.state = rolling_state(self.state, &self.request.surface, self.body)
            .map_err(|error| self.numerical(error.reason()))?;
        let relative = center_relative_velocity(self.state, &self.request.surface);
        if norm(relative) <= self.settings.velocity_tolerance_m_s
            && can_rest(self.state, &self.request.surface, self.body, self.settings)
        {
            return self.record_rest();
        }
        let motion = rolling_motion(
            self.state,
            &self.request.surface,
            self.body,
            self.settings.gravity_m_s2,
        );
        let stop = time_to_zero(
            relative,
            motion.acceleration,
            self.settings.velocity_tolerance_m_s,
        );
        let reaches_zero = stop.is_some_and(|time| time <= duration);
        self.advance(
            motion,
            stop.filter(|_| reaches_zero).unwrap_or(duration),
            is_cancelled,
        )?;
        if reaches_zero {
            self.state = rolling_state(self.state, &self.request.surface, self.body)
                .map_err(|error| self.numerical(error.reason()))?;
            if can_rest(self.state, &self.request.surface, self.body, self.settings) {
                return self.record_rest();
            }
        }
        Ok(None)
    }

    fn record_rest(
        &mut self,
    ) -> Result<Option<GroundTerminationReasonV1>, GroundReferenceRuntimeErrorV1> {
        let stopped = State {
            velocity: ZERO,
            spin: ZERO,
            ..self.state
        };
        if !self.append_event(GroundEventTypeV1::Rest, self.state, stopped)? {
            self.append_point(self.state, self.phase.wire())?;
            return Ok(Some(GroundTerminationReasonV1::EventLimit));
        }
        self.state = stopped;
        self.append_point(stopped, GroundPhaseV1::Rest)?;
        Ok(Some(GroundTerminationReasonV1::Rest))
    }

    fn advance(
        &mut self,
        motion: Motion,
        duration: f64,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), GroundReferenceRuntimeErrorV1> {
        let start = self.state;
        self.emit_grid_points(start, motion, duration, is_cancelled)?;
        self.state =
            advance(start, motion, duration).map_err(|error| self.numerical(error.reason()))?;
        self.ledger
            .record(LedgerStep {
                phase: self.phase,
                start,
                end: self.state,
                motion,
                duration,
                surface: &self.request.surface,
                body: self.body,
                gravity: self.settings.gravity_m_s2,
            })
            .map_err(|error| self.numerical(error.reason()))?;
        Ok(())
    }

    fn emit_grid_points(
        &mut self,
        start: State,
        motion: Motion,
        duration: f64,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<(), GroundReferenceRuntimeErrorV1> {
        let terminal = start.time + duration;
        self.output_schedule
            .skip_through(start.time, self.settings.time_tolerance_s)
            .ok_or_else(|| self.numerical("output_schedule"))?;
        loop {
            let sample_time = self
                .output_schedule
                .next_elapsed()
                .map_err(|error| self.numerical(error.reason()))?;
            if sample_time >= terminal - self.settings.time_tolerance_s {
                break;
            }
            if is_cancelled() {
                return Err(self.error(GroundReferenceRuntimeCodeV1::Cancelled, "cancelled"));
            }
            let sample = advance(start, motion, sample_time - start.time)
                .map_err(|error| self.numerical(error.reason()))?;
            self.append_point(sample, self.phase.wire())?;
            self.output_schedule
                .advance()
                .ok_or_else(|| self.numerical("output_schedule"))?;
        }
        Ok(())
    }

    fn append_point(
        &mut self,
        state: State,
        phase: GroundPhaseV1,
    ) -> Result<(), GroundReferenceRuntimeErrorV1> {
        if state.time <= self.handoff_time + self.settings.time_tolerance_s {
            return Ok(());
        }
        let point = state
            .point(&self.request.surface.frame, phase, self.time_origin)
            .map_err(|error| self.numerical(error.reason()))?;
        match self
            .timeline
            .classify(state.time, point.time_s, self.settings.time_tolerance_s)
        {
            Ok(TimelineAppend::Push) => {
                ensure_trajectory_capacity(
                    self.prefix_trajectory_count + self.trajectory.len(),
                    GroundReferencePhaseV1::SkidRoll,
                    self.fingerprint,
                )?;
                self.trajectory.push(point);
            }
            Ok(TimelineAppend::Replace) => {
                let Some(last) = self.trajectory.last_mut() else {
                    return Err(self.numerical("trajectory_order"));
                };
                *last = point;
            }
            Err(TimelineError::Order) => return Err(self.numerical("trajectory_order")),
            Err(TimelineError::Resolution) => {
                return Err(self.error(
                    GroundReferenceRuntimeCodeV1::ExecutionFailure,
                    "time_resolution",
                ))
            }
        }
        Ok(())
    }

    fn append_event(
        &mut self,
        event_type: GroundEventTypeV1,
        before: State,
        after: State,
    ) -> Result<bool, GroundReferenceRuntimeErrorV1> {
        SurfaceEventContext::new(
            self.request,
            self.time_origin,
            self.prefix_event_count,
            self.fingerprint,
        )
        .append(&mut self.events, event_type, before, after)
    }

    fn finish(
        self,
        termination: GroundTerminationReasonV1,
    ) -> Result<SurfaceSuffix, GroundReferenceRuntimeErrorV1> {
        self.ledger
            .validate_passivity(self.initial_state, self.state, self.body)
            .map_err(|reason| self.numerical(reason))?;
        Ok(SurfaceSuffix {
            trajectory: self.trajectory,
            events: self.events,
            state: self.state,
            skid_distance: self.ledger.skid_distance,
            roll_distance: self.ledger.roll_distance,
            termination,
        })
    }

    fn error(
        &self,
        code: GroundReferenceRuntimeCodeV1,
        reason: &str,
    ) -> GroundReferenceRuntimeErrorV1 {
        GroundReferenceRuntimeErrorV1::new(
            code,
            GroundReferencePhaseV1::SkidRoll,
            reason,
            self.fingerprint.to_owned(),
        )
    }

    fn numerical(&self, reason: &str) -> GroundReferenceRuntimeErrorV1 {
        self.error(GroundReferenceRuntimeCodeV1::NumericalFailure, reason)
    }
}
