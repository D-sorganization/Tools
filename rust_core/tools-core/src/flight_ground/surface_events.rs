//! Strict event append boundary for the compiled surface phase.

use super::execution_v1::{
    GroundReferencePhaseV1, GroundReferenceRuntimeCodeV1, GroundReferenceRuntimeErrorV1,
};
use super::request_v1::FlightToGroundRequestV1;
use super::resource_limits::ensure_event_capacity;
use super::result_v1::{GroundEventTypeV1, GroundEventV1};
use super::runtime_math::{wire_time, State};

pub(super) struct SurfaceEventContext<'a> {
    request: &'a FlightToGroundRequestV1,
    time_origin: f64,
    prefix_event_count: usize,
    fingerprint: &'a str,
}

impl<'a> SurfaceEventContext<'a> {
    pub fn new(
        request: &'a FlightToGroundRequestV1,
        time_origin: f64,
        prefix_event_count: usize,
        fingerprint: &'a str,
    ) -> Self {
        Self {
            request,
            time_origin,
            prefix_event_count,
            fingerprint,
        }
    }

    pub fn append(
        &self,
        events: &mut Vec<GroundEventV1>,
        event_type: GroundEventTypeV1,
        before: State,
        after: State,
    ) -> Result<bool, GroundReferenceRuntimeErrorV1> {
        let sequence = self.prefix_event_count + events.len();
        if sequence as u64 >= self.request.max_events {
            return Ok(false);
        }
        ensure_event_capacity(sequence, GroundReferencePhaseV1::SkidRoll, self.fingerprint)?;
        let time_s = wire_time(self.time_origin, before.time).map_err(|error| {
            GroundReferenceRuntimeErrorV1::new(
                GroundReferenceRuntimeCodeV1::NumericalFailure,
                GroundReferencePhaseV1::SkidRoll,
                error.reason(),
                self.fingerprint.to_owned(),
            )
        })?;
        events.push(GroundEventV1 {
            sequence: sequence as u64,
            event_type,
            time_s,
            frame: self.request.surface.frame.clone(),
            position_m: before.position,
            velocity_before_m_s: before.velocity,
            velocity_after_m_s: after.velocity,
            angular_velocity_before_rad_s: before.spin,
            angular_velocity_after_rad_s: after.spin,
        });
        Ok(true)
    }
}
