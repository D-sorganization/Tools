//! Deterministic repeated rigid impacts through the exact skid handoff.

use super::execution_v1::{
    BounceExecutionSettingsV1, GroundReferencePhaseV1, GroundReferenceRuntimeCodeV1,
    GroundReferenceRuntimeErrorV1,
};
use super::impact_runtime::{resolve_impact, Impact};
use super::request_v1::FlightToGroundRequestV1;
use super::resource_limits::{ensure_event_capacity, ensure_trajectory_capacity};
use super::result_v1::{GroundEventTypeV1, GroundEventV1, GroundPhaseV1, GroundTrajectoryPointV1};
use super::runtime_math::{
    ballistic, dot, ensure_numeric_range, interpolate_contact, project_contact, sub, wire_time,
    Body, MathResult, OutputSchedule, State, TimelineAppend, TimelineError, WireTimeline,
};

#[derive(Debug)]
pub(super) struct BouncePrefix {
    pub trajectory: Vec<GroundTrajectoryPointV1>,
    pub events: Vec<GroundEventV1>,
    pub handoff: State,
    pub bounce_air_distance: f64,
    pub time_origin: f64,
    pub output_schedule: OutputSchedule,
}

struct BounceRun<'a> {
    request: &'a FlightToGroundRequestV1,
    settings: &'a BounceExecutionSettingsV1,
    body: Body,
    fingerprint: &'a str,
    trajectory: Vec<GroundTrajectoryPointV1>,
    events: Vec<GroundEventV1>,
    bounce_air_distance: f64,
    time_origin: f64,
    output_schedule: OutputSchedule,
    timeline: WireTimeline,
}

pub(super) fn simulate_bounce<C>(
    request: &FlightToGroundRequestV1,
    settings: &BounceExecutionSettingsV1,
    fingerprint: &str,
    is_cancelled: &mut C,
) -> Result<BouncePrefix, GroundReferenceRuntimeErrorV1>
where
    C: FnMut() -> bool,
{
    if is_cancelled() {
        return Err(runtime_error(fingerprint, "cancelled"));
    }
    let mut contact = interpolate_contact(request)
        .map_err(|error| numerical_error(fingerprint, error.reason()))?;
    let time_origin = contact.time;
    contact.time = 0.0;
    let mut run = BounceRun::new(request, settings, fingerprint, time_origin);
    let first = run.resolve_and_record(contact, GroundEventTypeV1::FirstContact)?;
    run.after_impact(first, is_cancelled)
}

impl<'a> BounceRun<'a> {
    fn new(
        request: &'a FlightToGroundRequestV1,
        settings: &'a BounceExecutionSettingsV1,
        fingerprint: &'a str,
        time_origin: f64,
    ) -> Self {
        Self {
            request,
            settings,
            body: Body::from_request(request),
            fingerprint,
            trajectory: Vec::new(),
            events: Vec::new(),
            bounce_air_distance: 0.0,
            time_origin,
            output_schedule: OutputSchedule::new(request.output_interval_s),
            timeline: WireTimeline::default(),
        }
    }

    fn resolve_and_record(
        &mut self,
        incoming: State,
        event_type: GroundEventTypeV1,
    ) -> Result<Impact, GroundReferenceRuntimeErrorV1> {
        let speed = -self.relative_normal_speed(incoming);
        let restitution = if speed <= self.settings.capture_speed_m_s {
            0.0
        } else {
            self.request.surface.normal_restitution
        };
        let impact = resolve_impact(incoming, &self.request.surface, self.body, restitution)
            .map_err(|reason| self.numerical(reason))?;
        let phase = if event_type == GroundEventTypeV1::FirstContact {
            GroundPhaseV1::Impact
        } else {
            GroundPhaseV1::Bounce
        };
        self.append_state(impact.after, phase)?;
        let event = event(
            &impact,
            event_type,
            self.events.len() as u64,
            self.request,
            self.time_origin,
        )
        .map_err(|error| self.numerical(error.reason()))?;
        ensure_event_capacity(
            self.events.len(),
            GroundReferencePhaseV1::Bounce,
            self.fingerprint,
        )?;
        self.events.push(event);
        Ok(impact)
    }

    fn after_impact<C>(
        mut self,
        mut impact: Impact,
        is_cancelled: &mut C,
    ) -> Result<BouncePrefix, GroundReferenceRuntimeErrorV1>
    where
        C: FnMut() -> bool,
    {
        loop {
            let outgoing = impact.after;
            if impact.restitution == 0.0 {
                self.append_state(outgoing, GroundPhaseV1::Skid)?;
                return Ok(self.finish(outgoing));
            }
            if is_cancelled() {
                return Err(runtime_error(self.fingerprint, "cancelled"));
            }
            if self.events.len() as u64 >= self.request.max_events {
                return Err(self.failure("event_limit"));
            }
            let incoming = self
                .next_contact(outgoing)?
                .ok_or_else(|| self.failure("no_recontact"))?;
            let time_limit = self.request.max_time_s;
            if incoming.time > time_limit + self.settings.time_tolerance_s {
                return Err(self.failure("time_limit"));
            }
            self.sample_hop(outgoing, incoming, is_cancelled)?;
            impact = self.resolve_and_record(incoming, GroundEventTypeV1::Bounce)?;
        }
    }

    fn next_contact(
        &self,
        outgoing: State,
    ) -> Result<Option<State>, GroundReferenceRuntimeErrorV1> {
        let normal = self.request.surface.normal_unit;
        let normal_speed = self.relative_normal_speed(outgoing);
        let gravity_normal = dot(self.settings.gravity_m_s2, normal);
        if normal_speed <= self.settings.velocity_tolerance_m_s
            || gravity_normal >= -self.settings.velocity_tolerance_m_s
        {
            return Ok(None);
        }
        let duration = -2.0 * normal_speed / gravity_normal;
        if duration <= self.settings.time_tolerance_s {
            return Ok(None);
        }
        let airborne = ballistic(outgoing, self.settings.gravity_m_s2, duration)
            .map_err(|error| self.numerical(error.reason()))?;
        let contact = project_contact(airborne, self.request)
            .map_err(|error| self.numerical(error.reason()))?;
        Ok(Some(contact))
    }

    fn sample_hop<C>(
        &mut self,
        outgoing: State,
        incoming: State,
        is_cancelled: &mut C,
    ) -> Result<(), GroundReferenceRuntimeErrorV1>
    where
        C: FnMut() -> bool,
    {
        let tolerance = self.settings.time_tolerance_s;
        self.output_schedule
            .skip_through(outgoing.time, tolerance)
            .ok_or_else(|| self.numerical("output_schedule"))?;
        loop {
            let sample_time = self
                .output_schedule
                .next_elapsed()
                .map_err(|error| self.numerical(error.reason()))?;
            if sample_time >= incoming.time - tolerance {
                break;
            }
            if is_cancelled() {
                return Err(runtime_error(self.fingerprint, "cancelled"));
            }
            let sample = ballistic(
                outgoing,
                self.settings.gravity_m_s2,
                sample_time - outgoing.time,
            )
            .map_err(|error| self.numerical(error.reason()))?;
            self.append_state(sample, GroundPhaseV1::Bounce)?;
            self.output_schedule
                .advance()
                .ok_or_else(|| self.numerical("output_schedule"))?;
        }
        let terminal = ballistic(
            outgoing,
            self.settings.gravity_m_s2,
            incoming.time - outgoing.time,
        )
        .map_err(|error| self.numerical(error.reason()))?;
        let displacement = sub(terminal.position, outgoing.position);
        let distance = self.bounce_air_distance + displacement[0].hypot(displacement[2]);
        ensure_numeric_range(distance).map_err(|error| self.numerical(error.reason()))?;
        self.bounce_air_distance = distance;
        Ok(())
    }

    fn relative_normal_speed(&self, state: State) -> f64 {
        dot(
            sub(state.velocity, self.request.surface.surface_velocity_m_s),
            self.request.surface.normal_unit,
        )
    }

    fn append_state(
        &mut self,
        state: State,
        phase: GroundPhaseV1,
    ) -> Result<(), GroundReferenceRuntimeErrorV1> {
        let point = state
            .point(&self.request.surface.frame, phase, self.time_origin)
            .map_err(|error| self.numerical(error.reason()))?;
        match self
            .timeline
            .classify(state.time, point.time_s, self.settings.time_tolerance_s)
        {
            Ok(TimelineAppend::Push) => {
                ensure_trajectory_capacity(
                    self.trajectory.len(),
                    GroundReferencePhaseV1::Bounce,
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
            Err(TimelineError::Resolution) => return Err(self.failure("time_resolution")),
        }
        Ok(())
    }

    fn finish(self, handoff: State) -> BouncePrefix {
        BouncePrefix {
            trajectory: self.trajectory,
            events: self.events,
            handoff,
            bounce_air_distance: self.bounce_air_distance,
            time_origin: self.time_origin,
            output_schedule: self.output_schedule,
        }
    }

    fn failure(&self, reason: &str) -> GroundReferenceRuntimeErrorV1 {
        GroundReferenceRuntimeErrorV1::new(
            GroundReferenceRuntimeCodeV1::ExecutionFailure,
            GroundReferencePhaseV1::Bounce,
            reason,
            self.fingerprint.to_owned(),
        )
    }

    fn numerical(&self, reason: &str) -> GroundReferenceRuntimeErrorV1 {
        GroundReferenceRuntimeErrorV1::new(
            GroundReferenceRuntimeCodeV1::NumericalFailure,
            GroundReferencePhaseV1::Bounce,
            reason,
            self.fingerprint.to_owned(),
        )
    }
}

fn event(
    impact: &Impact,
    event_type: GroundEventTypeV1,
    sequence: u64,
    request: &FlightToGroundRequestV1,
    time_origin: f64,
) -> MathResult<GroundEventV1> {
    Ok(GroundEventV1 {
        sequence,
        event_type,
        time_s: wire_time(time_origin, impact.before.time)?,
        frame: request.surface.frame.clone(),
        position_m: impact.before.position,
        velocity_before_m_s: impact.before.velocity,
        velocity_after_m_s: impact.after.velocity,
        angular_velocity_before_rad_s: impact.before.spin,
        angular_velocity_after_rad_s: impact.after.spin,
    })
}

fn numerical_error(fingerprint: &str, reason: &str) -> GroundReferenceRuntimeErrorV1 {
    GroundReferenceRuntimeErrorV1::new(
        GroundReferenceRuntimeCodeV1::NumericalFailure,
        GroundReferencePhaseV1::Bounce,
        reason,
        fingerprint.to_owned(),
    )
}

fn runtime_error(fingerprint: &str, reason: &str) -> GroundReferenceRuntimeErrorV1 {
    GroundReferenceRuntimeErrorV1::new(
        GroundReferenceRuntimeCodeV1::Cancelled,
        GroundReferencePhaseV1::Bounce,
        reason,
        fingerprint.to_owned(),
    )
}
