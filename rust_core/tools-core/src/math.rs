//! Core mathematical utility functions.
//!
//! These are the **canonical implementations** for interpolation, clamping,
//! and angle conversions used across all downstream consumers.
//!
//! # Design by Contract
//! - All functions validate preconditions via `debug_assert!`.
//! - Functions return `Result<T, E>` when failure is possible.
//!
//! # DRY
//! - Python and WASM wrappers call these functions directly.

/// Linear interpolation between `a` and `b` by factor `t`.
///
/// # Contracts
/// - Precondition: `t` is in `[0.0, 1.0]` (debug-only; does not clamp in release).
/// - Postcondition: result is between `a` and `b` when `t ∈ [0, 1]`.
#[must_use]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    debug_assert!(!t.is_nan(), "lerp: interpolation factor t must not be NaN");
    debug_assert!(
        (0.0..=1.0).contains(&t),
        "lerp: t must be in [0.0, 1.0], got {t}"
    );
    a + (b - a) * t
}

/// Clamp a value to the range `[min_val, max_val]`.
///
/// # Contracts
/// - Precondition: `min_val <= max_val`.
/// - Postcondition: `min_val <= result <= max_val`.
#[must_use]
pub fn clamp(value: f64, min_val: f64, max_val: f64) -> f64 {
    debug_assert!(
        min_val <= max_val,
        "clamp: min_val ({min_val}) must be <= max_val ({max_val})"
    );
    if value < min_val {
        min_val
    } else if value > max_val {
        max_val
    } else {
        value
    }
}

/// Convert degrees to radians.
#[must_use]
pub fn deg_to_rad(degrees: f64) -> f64 {
    debug_assert!(!degrees.is_nan(), "deg_to_rad: degrees must not be NaN");
    degrees * std::f64::consts::PI / 180.0
}

/// Convert radians to degrees.
#[must_use]
pub fn rad_to_deg(radians: f64) -> f64 {
    debug_assert!(!radians.is_nan(), "rad_to_deg: radians must not be NaN");
    radians * 180.0 / std::f64::consts::PI
}

// ── Python bindings (feature-gated) ──────────────────────────────────────────

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "lerp")]
pub fn py_lerp(a: f64, b: f64, t: f64) -> f64 {
    lerp(a, b, t)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "clamp")]
pub fn py_clamp(value: f64, min_val: f64, max_val: f64) -> f64 {
    clamp(value, min_val, max_val)
}

// ── Tests (TDD — written before implementation was finalized) ────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── lerp ──

    #[test]
    fn test_lerp_at_zero() {
        assert!((lerp(10.0, 20.0, 0.0) - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lerp_at_one() {
        assert!((lerp(10.0, 20.0, 1.0) - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lerp_at_half() {
        assert!((lerp(10.0, 20.0, 0.5) - 15.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lerp_negative_range() {
        assert!((lerp(-10.0, 10.0, 0.5) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "t must be in [0.0, 1.0]")]
    fn test_lerp_t_out_of_range_panics() {
        let _r = lerp(0.0, 1.0, 1.5);
    }

    // ── clamp ──

    #[test]
    fn test_clamp_within_range() {
        assert!((clamp(5.0, 0.0, 10.0) - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clamp_below_min() {
        assert!((clamp(-5.0, 0.0, 10.0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_clamp_above_max() {
        assert!((clamp(15.0, 0.0, 10.0) - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "min_val")]
    fn test_clamp_invalid_range_panics() {
        let _r = clamp(5.0, 10.0, 0.0);
    }

    // ── Angle conversions ──

    #[test]
    fn test_deg_to_rad_90() {
        let r = deg_to_rad(90.0);
        assert!((r - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_deg_to_rad_180() {
        let r = deg_to_rad(180.0);
        assert!((r - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_rad_to_deg_pi() {
        let d = rad_to_deg(std::f64::consts::PI);
        assert!((d - 180.0).abs() < 1e-12);
    }

    #[test]
    fn test_deg_rad_roundtrip() {
        let original = 42.5;
        let result = rad_to_deg(deg_to_rad(original));
        assert!((result - original).abs() < 1e-12);
    }
}
