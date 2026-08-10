use serde::de::{Error as DeError, Visitor};
use serde::{Deserializer, Serializer};
use serde_json::Value;
use std::fmt;

use super::request_v1::{FlightToGroundRequestV1, GroundContactStateV1, GroundSurfaceV1};

const DECIMAL_SCALE: u128 = 100_000_000_000;
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MIN_POSITIVE: f64 = 1.0e-11;

pub(super) fn canonical_json(value: &Value) -> Result<String, ()> {
    match value {
        Value::Null => Ok("null".to_owned()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => canonical_number(value),
        Value::String(value) => serde_json::to_string(value).map_err(|_| ()),
        Value::Array(values) => canonical_array(values),
        Value::Object(values) => canonical_object(values),
    }
}

pub(super) fn normalize_f64(value: f64) -> Result<f64, ()> {
    canonical_f64_token(value)?.parse().map_err(|_| ())
}

pub(super) fn reject_unsafe_numbers(value: &Value) -> Result<(), ()> {
    match value {
        Value::Number(number) => reject_unsafe_number(number),
        Value::Array(values) => values.iter().try_for_each(reject_unsafe_numbers),
        Value::Object(values) => values.values().try_for_each(reject_unsafe_numbers),
        _ => Ok(()),
    }
}

pub(super) fn normalize_request(request: &mut FlightToGroundRequestV1) -> Result<(), ()> {
    normalize_scalar(&mut request.ball_radius_m)?;
    normalize_scalar(&mut request.ball_mass_kg)?;
    normalize_scalar(&mut request.rotational_inertia_factor)?;
    normalize_scalar(&mut request.max_time_s)?;
    normalize_scalar(&mut request.output_interval_s)?;
    normalize_surface(&mut request.surface)?;
    normalize_contact(&mut request.last_separated_state)?;
    normalize_contact(&mut request.first_penetrating_state)?;
    normalize_scalar(&mut request.calibration.confidence)
}

pub(super) fn validate_raw_numbers(request: &FlightToGroundRequestV1) -> Result<(), &'static str> {
    require_positive(request.ball_radius_m, "ball_radius_m")?;
    require_positive(request.ball_mass_kg, "ball_mass_kg")?;
    require_range(
        request.rotational_inertia_factor,
        MIN_POSITIVE,
        1.0,
        "rotational_inertia_factor",
    )?;
    require_positive(request.max_time_s, "max_time_s")?;
    require_positive(request.output_interval_s, "output_interval_s")?;
    if request.output_interval_s > request.max_time_s {
        return Err("solver_limits");
    }
    validate_raw_surface(&request.surface)?;
    validate_raw_contact(&request.last_separated_state)?;
    validate_raw_contact(&request.first_penetrating_state)?;
    require_range(
        request.calibration.confidence,
        0.0,
        1.0,
        "calibration_confidence",
    )
}

pub(super) fn deserialize_safe_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_any(SafeIntegerVisitor)
}

pub(super) fn serialize_safe_u64<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_u64(*value)
}

fn canonical_number(value: &serde_json::Number) -> Result<String, ()> {
    if let Some(integer) = value.as_i64() {
        return safe_i64(integer).map(|_| integer.to_string());
    }
    if let Some(integer) = value.as_u64() {
        return safe_u64(integer).map(|_| integer.to_string());
    }
    canonical_f64_token(value.as_f64().ok_or(())?)
}

fn normalize_surface(surface: &mut GroundSurfaceV1) -> Result<(), ()> {
    normalize_scalar(&mut surface.height_m)?;
    normalize_array(&mut surface.normal_unit)?;
    normalize_array(&mut surface.surface_velocity_m_s)?;
    for value in [
        &mut surface.normal_restitution,
        &mut surface.static_friction,
        &mut surface.kinetic_friction,
        &mut surface.rolling_resistance,
        &mut surface.firmness_pa,
        &mut surface.hardness_fraction,
        &mut surface.grass_height_m,
        &mut surface.compressibility_fraction,
        &mut surface.compression_damping_fraction,
        &mut surface.turf_density_kg_m3,
        &mut surface.moisture_fraction,
    ] {
        normalize_scalar(value)?;
    }
    Ok(())
}

fn normalize_contact(contact: &mut GroundContactStateV1) -> Result<(), ()> {
    normalize_scalar(&mut contact.time_s)?;
    normalize_array(&mut contact.position_m)?;
    normalize_array(&mut contact.velocity_m_s)?;
    normalize_array(&mut contact.angular_velocity_rad_s)
}

fn normalize_array(values: &mut [f64; 3]) -> Result<(), ()> {
    values.iter_mut().try_for_each(normalize_scalar)
}

fn normalize_scalar(value: &mut f64) -> Result<(), ()> {
    *value = normalize_f64(*value)?;
    Ok(())
}

fn validate_raw_surface(surface: &GroundSurfaceV1) -> Result<(), &'static str> {
    require_finite(surface.height_m, "surface_frame")?;
    validate_finite_array(surface.normal_unit, "surface_geometry")?;
    validate_finite_array(surface.surface_velocity_m_s, "surface_geometry")?;
    require_range(surface.normal_restitution, 0.0, 1.0, "normal_restitution")?;
    require_range(surface.static_friction, 0.0, 5.0, "static_friction")?;
    require_range(surface.kinetic_friction, 0.0, 5.0, "kinetic_friction")?;
    if surface.kinetic_friction > surface.static_friction {
        return Err("friction_order");
    }
    require_range(surface.rolling_resistance, 0.0, 1.0, "rolling_resistance")?;
    require_positive(surface.firmness_pa, "firmness_pa")?;
    require_range(surface.hardness_fraction, 0.0, 1.0, "hardness_fraction")?;
    require_range(surface.grass_height_m, 0.0, f64::MAX, "grass_height_m")?;
    validate_raw_surface_tail(surface)
}

fn validate_raw_surface_tail(surface: &GroundSurfaceV1) -> Result<(), &'static str> {
    require_range(
        surface.compressibility_fraction,
        0.0,
        1.0,
        "compressibility_fraction",
    )?;
    require_range(
        surface.compression_damping_fraction,
        0.0,
        1.0,
        "compression_damping_fraction",
    )?;
    require_range(
        surface.turf_density_kg_m3,
        0.0,
        f64::MAX,
        "turf_density_kg_m3",
    )?;
    require_range(surface.moisture_fraction, 0.0, 1.0, "moisture_fraction")
}

fn validate_raw_contact(contact: &GroundContactStateV1) -> Result<(), &'static str> {
    require_range(contact.time_s, 0.0, f64::MAX, "contact_state")?;
    validate_finite_array(contact.position_m, "contact_state_vector")?;
    validate_finite_array(contact.velocity_m_s, "contact_state_vector")?;
    validate_finite_array(contact.angular_velocity_rad_s, "contact_state_vector")
}

fn validate_finite_array(values: [f64; 3], field: &'static str) -> Result<(), &'static str> {
    values
        .into_iter()
        .try_for_each(|value| require_finite(value, field))
}

fn require_positive(value: f64, field: &'static str) -> Result<(), &'static str> {
    require_range(value, MIN_POSITIVE, f64::MAX, field)
}

fn require_finite(value: f64, field: &'static str) -> Result<(), &'static str> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(field)
    }
}

fn require_range(
    value: f64,
    lower: f64,
    upper: f64,
    field: &'static str,
) -> Result<(), &'static str> {
    if value.is_finite() && value >= lower && value <= upper {
        Ok(())
    } else {
        Err(field)
    }
}

fn canonical_array(values: &[Value]) -> Result<String, ()> {
    let tokens = values
        .iter()
        .map(canonical_json)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(format!("[{}]", tokens.join(",")))
}

fn canonical_object(values: &serde_json::Map<String, Value>) -> Result<String, ()> {
    let mut entries = values.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    let tokens = entries
        .into_iter()
        .map(|(key, value)| {
            let key = serde_json::to_string(key).map_err(|_| ())?;
            Ok(format!("{key}:{}", canonical_json(value)?))
        })
        .collect::<Result<Vec<_>, ()>>()?;
    Ok(format!("{{{}}}", tokens.join(",")))
}

fn canonical_f64_token(value: f64) -> Result<String, ()> {
    if !value.is_finite() || value.abs() > MAX_SAFE_INTEGER as f64 {
        return Err(());
    }
    let scaled = scaled_abs_half_away(value.abs())?;
    if scaled == 0 {
        return Ok("0".to_owned());
    }
    let integer = scaled / DECIMAL_SCALE;
    let fraction = scaled % DECIMAL_SCALE;
    let sign = if value.is_sign_negative() { "-" } else { "" };
    if fraction == 0 {
        return Ok(format!("{sign}{integer}"));
    }
    let fraction = format!("{fraction:011}").trim_end_matches('0').to_owned();
    Ok(format!("{sign}{integer}.{fraction}"))
}

fn scaled_abs_half_away(value: f64) -> Result<u128, ()> {
    let bits = value.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (mantissa, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
    };
    let numerator = u128::from(mantissa).checked_mul(DECIMAL_SCALE).ok_or(())?;
    if exponent >= 0 {
        return numerator.checked_shl(exponent as u32).ok_or(());
    }
    rounded_power_of_two_division(numerator, exponent.unsigned_abs())
}

fn rounded_power_of_two_division(numerator: u128, shift: u32) -> Result<u128, ()> {
    if shift >= 128 {
        return Ok(0);
    }
    let divisor = 1_u128.checked_shl(shift).ok_or(())?;
    let quotient = numerator / divisor;
    let remainder = numerator % divisor;
    Ok(quotient + u128::from(remainder >= divisor / 2))
}

fn reject_unsafe_number(value: &serde_json::Number) -> Result<(), ()> {
    if let Some(integer) = value.as_i64() {
        return safe_i64(integer);
    }
    if let Some(integer) = value.as_u64() {
        return safe_u64(integer);
    }
    let number = value.as_f64().ok_or(())?;
    if number.is_finite() && number.abs() <= MAX_SAFE_INTEGER as f64 {
        Ok(())
    } else {
        Err(())
    }
}

fn safe_i64(value: i64) -> Result<(), ()> {
    if value.unsigned_abs() <= MAX_SAFE_INTEGER {
        Ok(())
    } else {
        Err(())
    }
}

fn safe_u64(value: u64) -> Result<(), ()> {
    if value <= MAX_SAFE_INTEGER {
        Ok(())
    } else {
        Err(())
    }
}

struct SafeIntegerVisitor;

impl Visitor<'_> for SafeIntegerVisitor {
    type Value = u64;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a nonnegative cross-runtime safe integer")
    }

    fn visit_u64<E: DeError>(self, value: u64) -> Result<Self::Value, E> {
        safe_u64(value)
            .map(|_| value)
            .map_err(|_| E::custom("unsafe integer"))
    }

    fn visit_i64<E: DeError>(self, value: i64) -> Result<Self::Value, E> {
        let converted = u64::try_from(value).map_err(|_| E::custom("negative integer"))?;
        self.visit_u64(converted)
    }

    fn visit_f64<E: DeError>(self, value: f64) -> Result<Self::Value, E> {
        if !value.is_finite() || value.fract() != 0.0 || value < 0.0 {
            return Err(E::custom("expected integer-valued number"));
        }
        self.visit_u64(value as u64)
    }
}
