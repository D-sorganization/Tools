//! Ordering, ledger, summary, and terminal-state invariants for ground results.

use super::result_v1::{
    FlightToGroundResultV1, GroundEventTypeV1, GroundPhaseV1, GroundResultStatusV1,
    GroundResultV1Error, GroundTerminationReasonV1,
};

const ABSOLUTE_TOLERANCE: f64 = 1.0e-8;
const RELATIVE_TOLERANCE: f64 = 1.0e-10;
const TERMINATION_TIME_TOLERANCE: f64 = 1.0e-9;

pub(super) fn validate_result_geometry(
    result: &FlightToGroundResultV1,
) -> Result<(), GroundResultV1Error> {
    if matches!(
        result.status,
        GroundResultStatusV1::Failed | GroundResultStatusV1::Unavailable
    ) {
        return Ok(());
    }
    validate_trajectory_order(result)?;
    validate_event_order(result)?;
    validate_first_contact(result)?;
    validate_summary(result)?;
    validate_event_bounds(result)?;
    validate_terminal_state(result)
}

fn validate_trajectory_order(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    if result.trajectory.is_empty() {
        return invalid("trajectory");
    }
    for window in result.trajectory.windows(2) {
        if window[1].time_s <= window[0].time_s
            || !phase_transition(window[0].phase, window[1].phase)
        {
            return invalid("trajectory_order");
        }
    }
    Ok(())
}

fn validate_event_order(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    for (index, event) in result.events.iter().enumerate() {
        if event.sequence != index as u64 {
            return invalid("event_sequence");
        }
    }
    for window in result.events.windows(2) {
        if window[1].time_s < window[0].time_s
            || !event_transition(window[0].event_type, window[1].event_type)
        {
            return invalid("event_order");
        }
    }
    Ok(())
}

fn validate_first_contact(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    let point = result
        .trajectory
        .first()
        .ok_or_else(|| invalid_error("trajectory"))?;
    let event = result
        .events
        .first()
        .ok_or_else(|| invalid_error("events"))?;
    if event.event_type != GroundEventTypeV1::FirstContact
        || point.phase != GroundPhaseV1::Impact
        || !close(point.time_s, event.time_s)
        || !vector_close(&point.position_m, &event.position_m)
        || !vector_close(&point.velocity_m_s, &event.velocity_after_m_s)
        || !vector_close(
            &point.angular_velocity_rad_s,
            &event.angular_velocity_after_rad_s,
        )
    {
        return invalid("first_contact");
    }
    Ok(())
}

fn validate_summary(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    let summary = result
        .summary
        .as_ref()
        .ok_or_else(|| invalid_error("summary"))?;
    let first = result
        .trajectory
        .first()
        .ok_or_else(|| invalid_error("trajectory"))?;
    let final_point = result
        .trajectory
        .last()
        .ok_or_else(|| invalid_error("trajectory"))?;
    let expected = [
        first.position_m[0].hypot(first.position_m[2]),
        final_point.position_m[0],
        final_point.position_m[2],
    ];
    let actual = [
        summary.carry_distance_m,
        summary.final_downrange_m,
        summary.final_offline_m,
    ];
    let bounce_count = result
        .events
        .iter()
        .filter(|event| event.event_type == GroundEventTypeV1::Bounce)
        .count() as u64;
    if !vector_close(&expected, &actual)
        || !close(
            summary.total_distance_m,
            final_point.position_m[0].hypot(final_point.position_m[2]),
        )
        || !close(
            summary.surface_path_distance_m,
            summary.skid_distance_m + summary.roll_distance_m,
        )
        || summary.bounce_count != bounce_count
    {
        return invalid("summary_geometry");
    }
    Ok(())
}

fn validate_event_bounds(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    let first_time = result.trajectory.first().unwrap().time_s;
    let final_time = result.trajectory.last().unwrap().time_s;
    if result
        .events
        .iter()
        .any(|event| event.time_s < first_time || event.time_s > final_time)
    {
        return invalid("event_time_bounds");
    }
    if (result.termination.time_s - final_time).abs() > TERMINATION_TIME_TOLERANCE {
        return invalid("termination_time");
    }
    Ok(())
}

fn validate_terminal_state(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    let point = result.trajectory.last().unwrap();
    let event = result.events.last().unwrap();
    if result.status == GroundResultStatusV1::Partial {
        if point.phase == GroundPhaseV1::Rest
            || matches!(
                event.event_type,
                GroundEventTypeV1::Rest | GroundEventTypeV1::LeftSurface
            )
        {
            return invalid("partial_terminal_state");
        }
        return Ok(());
    }
    let expected_event = match result.termination.reason {
        GroundTerminationReasonV1::Rest => GroundEventTypeV1::Rest,
        GroundTerminationReasonV1::LeftSurface => GroundEventTypeV1::LeftSurface,
        _ => return Err(GroundResultV1Error::StatusPayload),
    };
    if event.event_type != expected_event
        || !close(event.time_s, result.termination.time_s)
        || !vector_close(&event.position_m, &point.position_m)
        || !vector_close(&event.velocity_after_m_s, &point.velocity_m_s)
        || !vector_close(
            &event.angular_velocity_after_rad_s,
            &point.angular_velocity_rad_s,
        )
        || ((expected_event == GroundEventTypeV1::Rest) != (point.phase == GroundPhaseV1::Rest))
    {
        return invalid("terminal_state");
    }
    Ok(())
}

fn phase_transition(left: GroundPhaseV1, right: GroundPhaseV1) -> bool {
    match left {
        GroundPhaseV1::Impact => true,
        GroundPhaseV1::Bounce => !matches!(right, GroundPhaseV1::Impact),
        GroundPhaseV1::Skid => matches!(
            right,
            GroundPhaseV1::Skid | GroundPhaseV1::Roll | GroundPhaseV1::Rest
        ),
        GroundPhaseV1::Roll => matches!(right, GroundPhaseV1::Roll | GroundPhaseV1::Rest),
        GroundPhaseV1::Rest => right == GroundPhaseV1::Rest,
    }
}

fn event_transition(left: GroundEventTypeV1, right: GroundEventTypeV1) -> bool {
    match left {
        GroundEventTypeV1::FirstContact | GroundEventTypeV1::Bounce => matches!(
            right,
            GroundEventTypeV1::Bounce
                | GroundEventTypeV1::SkidToRoll
                | GroundEventTypeV1::Rest
                | GroundEventTypeV1::LeftSurface
        ),
        GroundEventTypeV1::SkidToRoll => {
            matches!(
                right,
                GroundEventTypeV1::Rest | GroundEventTypeV1::LeftSurface
            )
        }
        GroundEventTypeV1::Rest | GroundEventTypeV1::LeftSurface => false,
    }
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= ABSOLUTE_TOLERANCE.max(RELATIVE_TOLERANCE * left.abs().max(right.abs()))
}

fn vector_close(left: &[f64; 3], right: &[f64; 3]) -> bool {
    left.iter()
        .zip(right)
        .all(|(left_value, right_value)| close(*left_value, *right_value))
}

fn invalid<T>(field: &'static str) -> Result<T, GroundResultV1Error> {
    Err(invalid_error(field))
}

const fn invalid_error(field: &'static str) -> GroundResultV1Error {
    GroundResultV1Error::InvalidField(field)
}
