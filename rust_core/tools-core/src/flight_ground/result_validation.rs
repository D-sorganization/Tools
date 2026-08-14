//! Field and status validation for the strict ground-result wire contract.

use std::collections::HashSet;

use super::request_v1::{TARGET_FRAME, UNIT_SYSTEM};
use super::result_geometry::validate_result_geometry;
use super::result_v1::{
    FlightToGroundResultV1, GroundEventTypeV1, GroundPhaseV1, GroundResultStatusV1,
    GroundResultV1Error, GroundTerminationReasonV1, RESULT_SCHEMA_VERSION,
};

const REST_TOLERANCE: f64 = 1.0e-9;

pub(super) fn validate_result(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    validate_identity(result)?;
    validate_calibration(result)?;
    validate_provenance(result)?;
    validate_trajectory_fields(result)?;
    validate_event_fields(result)?;
    validate_summary_fields(result)?;
    validate_termination(result)?;
    validate_evidence_fields(result)?;
    validate_status_payload(result)?;
    validate_result_geometry(result)
}

fn validate_identity(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    if result.schema_version != RESULT_SCHEMA_VERSION
        || result.unit_system != UNIT_SYSTEM
        || result.frame != TARGET_FRAME
    {
        return Err(GroundResultV1Error::InvalidSchema);
    }
    for value in [
        &result.request_id,
        &result.surface_id,
        &result.model_id,
        &result.model_version,
    ] {
        validate_text(value, "result_identity")?;
    }
    Ok(())
}

fn validate_calibration(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    let calibration = &result.calibration;
    validate_text(&calibration.calibration_id, "calibration_id")?;
    validate_text(&calibration.source, "calibration_source")?;
    if !matches!(
        calibration.kind.as_str(),
        "measured" | "literature" | "estimated" | "unvalidated"
    ) || !in_range(calibration.confidence, 0.0, 1.0)
    {
        return invalid("calibration");
    }
    Ok(())
}

fn validate_provenance(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    let provenance = &result.provenance;
    for value in [
        &provenance.producer,
        &provenance.producer_version,
        &provenance.source_revision,
    ] {
        validate_text(value, "provenance")?;
    }
    let digest = provenance.input_sha256.as_bytes();
    if digest.len() != 64 || !digest.iter().all(u8::is_ascii_hexdigit) {
        return invalid("input_sha256");
    }
    Ok(())
}

fn validate_trajectory_fields(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    for point in &result.trajectory {
        if !nonnegative(point.time_s) || point.frame != result.frame {
            return invalid("trajectory_point");
        }
        validate_vectors(&[
            point.position_m,
            point.velocity_m_s,
            point.angular_velocity_rad_s,
        ])?;
        if point.phase == GroundPhaseV1::Rest
            && moving(&point.velocity_m_s, &point.angular_velocity_rad_s)
        {
            return invalid("rest_point");
        }
    }
    Ok(())
}

fn validate_event_fields(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    for event in &result.events {
        if !nonnegative(event.time_s) || event.frame != result.frame {
            return invalid("ground_event");
        }
        validate_vectors(&[
            event.position_m,
            event.velocity_before_m_s,
            event.velocity_after_m_s,
            event.angular_velocity_before_rad_s,
            event.angular_velocity_after_rad_s,
        ])?;
        if event.event_type == GroundEventTypeV1::Rest
            && moving(
                &event.velocity_after_m_s,
                &event.angular_velocity_after_rad_s,
            )
        {
            return invalid("rest_event");
        }
    }
    Ok(())
}

fn validate_summary_fields(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    let Some(summary) = &result.summary else {
        return Ok(());
    };
    let nonnegative_values = [
        summary.carry_distance_m,
        summary.bounce_air_distance_m,
        summary.skid_distance_m,
        summary.roll_distance_m,
        summary.surface_path_distance_m,
        summary.total_distance_m,
    ];
    if nonnegative_values
        .into_iter()
        .any(|value| !nonnegative(value))
        || !summary.final_downrange_m.is_finite()
        || !summary.final_offline_m.is_finite()
    {
        return invalid("summary");
    }
    Ok(())
}

fn validate_termination(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    let termination = &result.termination;
    let completed_reason = matches!(
        termination.reason,
        GroundTerminationReasonV1::Rest | GroundTerminationReasonV1::LeftSurface
    );
    if !nonnegative(termination.time_s) || termination.completed != completed_reason {
        return invalid("termination");
    }
    Ok(())
}

fn validate_evidence_fields(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    for warning in &result.warnings {
        validate_text(&warning.code, "warning")?;
        validate_text(&warning.message, "warning")?;
    }
    let mut field_ids = HashSet::new();
    for field in &result.unavailable_fields {
        validate_text(&field.provenance, "unavailable_field")?;
        if !field_ids.insert(field.field_id) {
            return invalid("unavailable_field_ids");
        }
    }
    Ok(())
}

fn validate_status_payload(result: &FlightToGroundResultV1) -> Result<(), GroundResultV1Error> {
    if !status_reason_matches(result.status, result.termination.reason) {
        return Err(GroundResultV1Error::StatusPayload);
    }
    let unavailable = result.status == GroundResultStatusV1::Unavailable;
    if unavailable == result.unavailable_fields.is_empty() {
        return Err(GroundResultV1Error::StatusPayload);
    }
    if matches!(
        result.status,
        GroundResultStatusV1::Failed | GroundResultStatusV1::Unavailable
    ) && (!result.trajectory.is_empty()
        || !result.events.is_empty()
        || result.summary.is_some()
        || result.termination.completed)
    {
        return Err(GroundResultV1Error::StatusPayload);
    }
    Ok(())
}

fn status_reason_matches(status: GroundResultStatusV1, reason: GroundTerminationReasonV1) -> bool {
    matches!(
        (status, reason),
        (
            GroundResultStatusV1::Complete,
            GroundTerminationReasonV1::Rest
        ) | (
            GroundResultStatusV1::Complete,
            GroundTerminationReasonV1::LeftSurface
        ) | (
            GroundResultStatusV1::Partial,
            GroundTerminationReasonV1::TimeLimit
        ) | (
            GroundResultStatusV1::Partial,
            GroundTerminationReasonV1::EventLimit
        ) | (
            GroundResultStatusV1::Failed,
            GroundTerminationReasonV1::NumericalFailure
        ) | (
            GroundResultStatusV1::Unavailable,
            GroundTerminationReasonV1::UnavailableInput
        )
    )
}

fn validate_text(value: &str, field: &'static str) -> Result<(), GroundResultV1Error> {
    const EDGE_WHITESPACE: &[char] = &[' ', '\t', '\r', '\n', '\u{000c}', '\u{000b}'];
    if value.is_empty() || value.trim_matches(EDGE_WHITESPACE) != value {
        return invalid(field);
    }
    Ok(())
}

fn validate_vectors(values: &[[f64; 3]]) -> Result<(), GroundResultV1Error> {
    if values.iter().flatten().any(|value| !value.is_finite()) {
        return invalid("result_vector");
    }
    Ok(())
}

fn moving(linear: &[f64; 3], angular: &[f64; 3]) -> bool {
    linear
        .iter()
        .chain(angular)
        .any(|value| value.abs() > REST_TOLERANCE)
}

fn nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn in_range(value: f64, lower: f64, upper: f64) -> bool {
    value.is_finite() && value >= lower && value <= upper
}

fn invalid<T>(field: &'static str) -> Result<T, GroundResultV1Error> {
    Err(GroundResultV1Error::InvalidField(field))
}
