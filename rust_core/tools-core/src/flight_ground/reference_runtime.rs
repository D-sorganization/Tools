//! One-shot compiled reference execution over bounce, skid, roll, and rest.

use sha2::{Digest, Sha256};

use super::bounce_runtime::{simulate_bounce, BouncePrefix};
use super::canonical_json::{canonical_json, scaled_abs_half_away};
use super::execution_v1::{
    GroundReferenceExecutionV1, GroundReferencePhaseV1, GroundReferenceRuntimeCodeV1,
    GroundReferenceRuntimeErrorV1,
};
use super::request_v1::{
    canonical_normalized_request_v1_json, normalized_request_v1, FlightToGroundRequestV1,
};
use super::resource_limits::validate_resource_limits;
use super::result_v1::{
    FlightToGroundResultV1, GroundPhaseV1, GroundResultStatusV1, GroundSummaryV1,
    GroundTerminationReasonV1, GroundTerminationV1, GroundTrajectoryPointV1,
    GroundWarningSeverityV1, GroundWarningV1, RESULT_SCHEMA_VERSION,
};
use super::runtime_math::{canonical_scalar, interpolate_contact, wire_time};
use super::surface_runtime::{simulate_surface, SurfaceSuffix};

const IMPACT_LIMITATION: &str = "Rigid restitution v1 does not use firmness_pa, hardness_fraction, grass_height_m, compressibility_fraction, compression_damping_fraction, turf_density_kg_m3, moisture_fraction, or rolling_resistance.";
const STATIC_PLANE_NOTICE: &str = "Qualified for one immutable planar profile; material regions and changing normals are unsupported.";
const AXIAL_SPIN_NOTICE: &str =
    "Normal-axis spin is preserved because v1 has no calibrated torsional damping law.";
const CENSORED_NOTICE: &str =
    "Distance totals describe only the observed endpoint and are not projected final-rest metrics.";

pub fn run_ground_reference_v1<C>(
    request: &FlightToGroundRequestV1,
    execution: &GroundReferenceExecutionV1,
    is_cancelled: C,
) -> Result<FlightToGroundResultV1, GroundReferenceRuntimeErrorV1>
where
    C: FnMut() -> bool,
{
    let normalized = normalized_request_v1(request).map_err(request_error)?;
    run_normalized_ground_reference_v1(&normalized, execution, is_cancelled)
}

pub(super) fn run_normalized_ground_reference_v1<C>(
    request: &FlightToGroundRequestV1,
    execution: &GroundReferenceExecutionV1,
    mut is_cancelled: C,
) -> Result<FlightToGroundResultV1, GroundReferenceRuntimeErrorV1>
where
    C: FnMut() -> bool,
{
    let fingerprint = request_fingerprint(request)?;
    execution.validate().map_err(|error| {
        GroundReferenceRuntimeErrorV1::new(
            GroundReferenceRuntimeCodeV1::ExecutionFailure,
            GroundReferencePhaseV1::Bounce,
            error.code(),
            fingerprint.clone(),
        )
    })?;
    validate_wire_time_resolution(request, &fingerprint)?;
    validate_resource_limits(request, execution, &fingerprint)?;
    let prefix = simulate_bounce(
        request,
        &execution.bounce_settings,
        &fingerprint,
        &mut is_cancelled,
    )?;
    let suffix = simulate_surface(
        request,
        &execution.skid_roll_settings,
        &prefix,
        &fingerprint,
        &mut is_cancelled,
    )?;
    if is_cancelled() {
        return Err(GroundReferenceRuntimeErrorV1::new(
            GroundReferenceRuntimeCodeV1::Cancelled,
            GroundReferencePhaseV1::Composition,
            "cancelled",
            fingerprint,
        ));
    }
    compose_result(request, prefix, suffix, fingerprint)
}

fn validate_wire_time_resolution(
    request: &FlightToGroundRequestV1,
    fingerprint: &str,
) -> Result<(), GroundReferenceRuntimeErrorV1> {
    let invalid = || {
        GroundReferenceRuntimeErrorV1::new(
            GroundReferenceRuntimeCodeV1::ExecutionFailure,
            GroundReferencePhaseV1::Bounce,
            "time_resolution",
            fingerprint.to_owned(),
        )
    };
    let contact = interpolate_contact(request).map_err(|error| {
        GroundReferenceRuntimeErrorV1::new(
            GroundReferenceRuntimeCodeV1::NumericalFailure,
            GroundReferencePhaseV1::Bounce,
            error.reason(),
            fingerprint.to_owned(),
        )
    })?;
    let origin = wire_time(contact.time, 0.0).map_err(|_| invalid())?;
    let first = wire_time(origin, request.output_interval_s).map_err(|_| invalid())?;
    if first <= origin {
        return Err(invalid());
    }

    let duration = scaled_abs_half_away(request.max_time_s).map_err(|_| invalid())?;
    let interval = scaled_abs_half_away(request.output_interval_s).map_err(|_| invalid())?;
    if interval == 0 {
        return Err(invalid());
    }
    let quotient = duration / interval;
    let previous_index = quotient.saturating_sub(u128::from(duration % interval == 0));
    let previous_elapsed = request.output_interval_s * previous_index as f64;
    let terminal = wire_time(origin, request.max_time_s).map_err(|_| invalid())?;
    let previous = wire_time(origin, previous_elapsed).map_err(|_| invalid())?;
    if terminal <= previous {
        return Err(invalid());
    }
    Ok(())
}

pub fn canonical_ground_reference_runtime_error_v1_json(
    error: &GroundReferenceRuntimeErrorV1,
) -> String {
    let value = serde_json::to_value(error).expect("runtime error fields are always serializable");
    canonical_json(&value).expect("runtime error fields contain no numeric values")
}

fn request_fingerprint(
    request: &FlightToGroundRequestV1,
) -> Result<String, GroundReferenceRuntimeErrorV1> {
    let canonical = canonical_normalized_request_v1_json(request).map_err(request_error)?;
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

fn request_error(error: super::GroundRequestV1Error) -> GroundReferenceRuntimeErrorV1 {
    GroundReferenceRuntimeErrorV1::new(
        GroundReferenceRuntimeCodeV1::ExecutionFailure,
        GroundReferencePhaseV1::Bounce,
        error.code(),
        "0".repeat(64),
    )
}

fn compose_result(
    request: &FlightToGroundRequestV1,
    mut prefix: BouncePrefix,
    mut suffix: SurfaceSuffix,
    fingerprint: String,
) -> Result<FlightToGroundResultV1, GroundReferenceRuntimeErrorV1> {
    let composition_error = || {
        GroundReferenceRuntimeErrorV1::new(
            GroundReferenceRuntimeCodeV1::ExecutionFailure,
            GroundReferencePhaseV1::Composition,
            "composition_error",
            fingerprint.clone(),
        )
    };
    let numeric_error = || {
        GroundReferenceRuntimeErrorV1::new(
            GroundReferenceRuntimeCodeV1::NumericalFailure,
            GroundReferencePhaseV1::Composition,
            "numeric_range",
            fingerprint.clone(),
        )
    };
    qualify_prefix_start(&mut prefix).map_err(|_| composition_error())?;
    if suffix.trajectory.is_empty()
        && !matches!(
            suffix.termination,
            GroundTerminationReasonV1::EventLimit | GroundTerminationReasonV1::TimeLimit
        )
    {
        return Err(composition_error());
    }
    let first = prefix.trajectory.first().ok_or_else(composition_error)?;
    let final_point = suffix
        .trajectory
        .last()
        .or_else(|| prefix.trajectory.last())
        .ok_or_else(composition_error)?;
    let status = if suffix.termination == GroundTerminationReasonV1::Rest {
        GroundResultStatusV1::Complete
    } else {
        GroundResultStatusV1::Partial
    };
    let carry_distance = canonical_scalar(first.position_m[0].hypot(first.position_m[2]))
        .map_err(|_| numeric_error())?;
    let surface_path_distance = canonical_scalar(suffix.skid_distance + suffix.roll_distance)
        .map_err(|_| numeric_error())?;
    let total_distance =
        canonical_scalar(final_point.position_m[0].hypot(final_point.position_m[2]))
            .map_err(|_| numeric_error())?;
    let summary = GroundSummaryV1 {
        carry_distance_m: carry_distance,
        bounce_air_distance_m: prefix.bounce_air_distance,
        skid_distance_m: suffix.skid_distance,
        roll_distance_m: suffix.roll_distance,
        surface_path_distance_m: surface_path_distance,
        total_distance_m: total_distance,
        final_downrange_m: final_point.position_m[0],
        final_offline_m: final_point.position_m[2],
        bounce_count: prefix
            .events
            .iter()
            .filter(|event| event.event_type == super::result_v1::GroundEventTypeV1::Bounce)
            .count() as u64,
    };
    let termination_time = wire_time(prefix.time_origin, suffix.state.time).map_err(|error| {
        GroundReferenceRuntimeErrorV1::new(
            GroundReferenceRuntimeCodeV1::NumericalFailure,
            GroundReferencePhaseV1::Composition,
            error.reason(),
            fingerprint.clone(),
        )
    })?;
    if final_point.time_s != termination_time {
        return Err(composition_error());
    }
    let termination = GroundTerminationV1 {
        reason: suffix.termination,
        time_s: termination_time,
        completed: status == GroundResultStatusV1::Complete,
    };
    prefix.trajectory.append(&mut suffix.trajectory);
    prefix.events.append(&mut suffix.events);
    let result = FlightToGroundResultV1 {
        schema_version: RESULT_SCHEMA_VERSION.to_owned(),
        request_id: request.request_id.clone(),
        surface_id: request.surface.surface_id.clone(),
        frame: request.surface.frame.clone(),
        model_id: "tools-ground-impact-bounce+tools-ground-skid-roll".to_owned(),
        model_version: "1.0.0+1.0.0".to_owned(),
        status,
        trajectory: prefix.trajectory,
        events: prefix.events,
        summary: Some(summary),
        termination,
        calibration: request.calibration.clone(),
        warnings: warnings(status),
        unavailable_fields: Vec::new(),
        provenance: request.provenance.clone(),
        unit_system: request.unit_system.clone(),
    };
    let value = serde_json::to_value(&result).map_err(|_| composition_error())?;
    canonical_json(&value).map_err(|_| numeric_error())?;
    Ok(result)
}

fn qualify_prefix_start(prefix: &mut BouncePrefix) -> Result<(), ()> {
    let Some(first) = prefix.trajectory.first() else {
        return Err(());
    };
    if first.phase == GroundPhaseV1::Impact {
        return Ok(());
    }
    if prefix.trajectory.len() != 1 || first.phase != GroundPhaseV1::Skid {
        return Err(());
    }
    let event = prefix.events.first().ok_or(())?;
    let reconstructed = GroundTrajectoryPointV1 {
        time_s: event.time_s,
        frame: event.frame.clone(),
        position_m: event.position_m,
        velocity_m_s: event.velocity_after_m_s,
        angular_velocity_rad_s: event.angular_velocity_after_rad_s,
        phase: GroundPhaseV1::Impact,
    };
    if reconstructed.time_s != first.time_s
        || reconstructed.position_m != first.position_m
        || reconstructed.velocity_m_s != first.velocity_m_s
        || reconstructed.angular_velocity_rad_s != first.angular_velocity_rad_s
    {
        return Err(());
    }
    prefix.trajectory[0] = reconstructed;
    Ok(())
}

fn warnings(status: GroundResultStatusV1) -> Vec<GroundWarningV1> {
    let mut warnings = vec![
        warning(
            "IMPACT_PREFIX_LIMITATION_001",
            IMPACT_LIMITATION,
            GroundWarningSeverityV1::Info,
        ),
        warning(
            "STATIC_PLANE_V1",
            STATIC_PLANE_NOTICE,
            GroundWarningSeverityV1::Info,
        ),
        warning(
            "AXIAL_SPIN_UNDAMPED",
            AXIAL_SPIN_NOTICE,
            GroundWarningSeverityV1::Info,
        ),
    ];
    if status != GroundResultStatusV1::Complete {
        warnings.push(warning(
            "CENSORED_ENDPOINT",
            CENSORED_NOTICE,
            GroundWarningSeverityV1::Warning,
        ));
    }
    warnings
}

fn warning(code: &str, message: &str, severity: GroundWarningSeverityV1) -> GroundWarningV1 {
    GroundWarningV1 {
        code: code.to_owned(),
        message: message.to_owned(),
        severity,
    }
}
