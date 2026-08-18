//! Independent resource budgets for synchronous compiled ground execution.

use super::canonical_json::scaled_abs_half_away;
use super::execution_v1::{
    GroundReferenceExecutionV1, GroundReferencePhaseV1, GroundReferenceRuntimeCodeV1,
    GroundReferenceRuntimeErrorV1,
};
use super::request_v1::FlightToGroundRequestV1;

const MAX_SCHEDULED_OUTPUT_POINTS: u128 = 200_001;
const MAX_INTEGRATION_STEPS: u64 = 1_000_000;
const MAX_EVENT_COUNT: u64 = 10_000;
const TERMINAL_PHASE_ALLOWANCE: u128 = 2;
const MAX_TRAJECTORY_POINTS: usize = 210_003;

pub(super) fn validate_resource_limits(
    request: &FlightToGroundRequestV1,
    execution: &GroundReferenceExecutionV1,
    fingerprint: &str,
) -> Result<(), GroundReferenceRuntimeErrorV1> {
    let duration = scaled_abs_half_away(request.max_time_s).map_err(|_| {
        failure(
            GroundReferencePhaseV1::Bounce,
            "output_point_limit",
            fingerprint,
        )
    })?;
    let output_interval = scaled_abs_half_away(request.output_interval_s).map_err(|_| {
        failure(
            GroundReferencePhaseV1::Bounce,
            "output_point_limit",
            fingerprint,
        )
    })?;
    let scheduled_points =
        endpoint_inclusive_count(duration, output_interval).ok_or_else(|| {
            failure(
                GroundReferencePhaseV1::Bounce,
                "output_point_limit",
                fingerprint,
            )
        })?;
    if scheduled_points > MAX_SCHEDULED_OUTPUT_POINTS {
        return Err(failure(
            GroundReferencePhaseV1::Bounce,
            "output_point_limit",
            fingerprint,
        ));
    }

    if execution.skid_roll_settings.max_steps > MAX_INTEGRATION_STEPS {
        return Err(failure(
            GroundReferencePhaseV1::SkidRoll,
            "integration_step_limit",
            fingerprint,
        ));
    }
    if request.max_events > MAX_EVENT_COUNT {
        return Err(failure(
            GroundReferencePhaseV1::Bounce,
            "event_count_limit",
            fingerprint,
        ));
    }
    let trajectory_budget = scheduled_points
        .checked_add(u128::from(request.max_events))
        .and_then(|count| count.checked_add(TERMINAL_PHASE_ALLOWANCE))
        .ok_or_else(|| {
            failure(
                GroundReferencePhaseV1::Bounce,
                "trajectory_point_limit",
                fingerprint,
            )
        })?;
    if trajectory_budget > MAX_TRAJECTORY_POINTS as u128 {
        return Err(failure(
            GroundReferencePhaseV1::Bounce,
            "trajectory_point_limit",
            fingerprint,
        ));
    }
    Ok(())
}

pub(super) fn ensure_trajectory_capacity(
    current_points: usize,
    phase: GroundReferencePhaseV1,
    fingerprint: &str,
) -> Result<(), GroundReferenceRuntimeErrorV1> {
    if current_points >= MAX_TRAJECTORY_POINTS {
        return Err(failure(phase, "trajectory_point_limit", fingerprint));
    }
    Ok(())
}

pub(super) fn ensure_event_capacity(
    current_events: usize,
    phase: GroundReferencePhaseV1,
    fingerprint: &str,
) -> Result<(), GroundReferenceRuntimeErrorV1> {
    if current_events >= MAX_EVENT_COUNT as usize {
        return Err(failure(phase, "event_count_limit", fingerprint));
    }
    Ok(())
}

fn endpoint_inclusive_count(duration: u128, interval: u128) -> Option<u128> {
    ceiling_ratio(duration, interval)?.checked_add(1)
}

fn ceiling_ratio(numerator: u128, denominator: u128) -> Option<u128> {
    if denominator == 0 {
        return None;
    }
    (numerator / denominator).checked_add(u128::from(!numerator.is_multiple_of(denominator)))
}

fn failure(
    phase: GroundReferencePhaseV1,
    reason: &str,
    fingerprint: &str,
) -> GroundReferenceRuntimeErrorV1 {
    GroundReferenceRuntimeErrorV1::new(
        GroundReferenceRuntimeCodeV1::ExecutionFailure,
        phase,
        reason,
        fingerprint.to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_trajectory_cap_accepts_boundary_and_rejects_next_push() {
        assert!(ensure_trajectory_capacity(
            MAX_TRAJECTORY_POINTS - 1,
            GroundReferencePhaseV1::Composition,
            "a"
        )
        .is_ok());
        let error = ensure_trajectory_capacity(
            MAX_TRAJECTORY_POINTS,
            GroundReferencePhaseV1::Composition,
            "a",
        )
        .unwrap_err();
        assert_eq!(error.native_reason, "trajectory_point_limit");
    }

    #[test]
    fn dynamic_event_cap_accepts_boundary_and_rejects_next_push() {
        assert!(ensure_event_capacity(
            MAX_EVENT_COUNT as usize - 1,
            GroundReferencePhaseV1::Bounce,
            "a"
        )
        .is_ok());
        let error = ensure_event_capacity(
            MAX_EVENT_COUNT as usize,
            GroundReferencePhaseV1::Bounce,
            "a",
        )
        .unwrap_err();
        assert_eq!(error.native_reason, "event_count_limit");
    }
}
