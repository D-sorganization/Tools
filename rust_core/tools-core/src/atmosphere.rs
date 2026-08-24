//! Standard atmosphere model (ISA — International Standard Atmosphere).
//!
//! Provides canonical pressure, density, temperature, and viscosity
//! calculations for altitudes in the troposphere (0–11 km).
//!
//! # Design by Contract
//! - Altitude must be finite and non-negative
//! - Output density, pressure, temperature are always positive
//!
//! # DRY
//! - This is the **single source of truth**. Downstream crates (upstream-physics,
//!   Gasification_Model) import these functions instead of defining local copies.

use crate::math::{GRAVITY, R_GAS};

// ── ISA constants (troposphere) ──────────────────────────────────────────────

/// Sea-level temperature [K].
const T0: f64 = 288.15;

/// Sea-level pressure [Pa].
const P0: f64 = 101_325.0;

/// Temperature lapse rate [K/m].
const LAPSE_RATE: f64 = 0.0065;

/// Molar mass of dry air [kg/mol].
const M_AIR: f64 = 0.0289644;

/// Reference viscosity at sea level [Pa·s].
const MU_REF: f64 = 1.81e-5;

/// Viscosity power-law exponent (Sutherland approximation, simplified).
const MU_EXPONENT: f64 = 0.76;

// ── Data Types ───────────────────────────────────────────────────────────────

/// Atmospheric properties at a given altitude.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct AtmosphereProperties {
    /// Air density [kg/m³].
    pub density: f64,
    /// Dynamic viscosity [Pa·s].
    pub viscosity: f64,
    /// Temperature [K].
    pub temperature: f64,
    /// Pressure [Pa].
    pub pressure: f64,
}

// ── Core Functions ───────────────────────────────────────────────────────────

/// Compute atmospheric properties at a given altitude using the ISA model.
///
/// Implements the International Standard Atmosphere for the troposphere
/// (0–11,000 m). Above the tropopause, results are approximate.
///
/// # Contracts (DbC)
/// - Precondition: `altitude_m` is finite and non-negative
/// - Postcondition: `density > 0`, `pressure > 0`, `temperature > 0`
///
/// # Arguments
/// * `altitude_m` — Altitude above sea level [m]
///
/// # Returns
/// `AtmosphereProperties` with density, viscosity, temperature, pressure
#[must_use]
pub fn atmosphere_at_altitude(altitude_m: f64) -> AtmosphereProperties {
    assert!(altitude_m.is_finite(), "DbC: altitude must be finite");
    assert!(altitude_m >= 0.0, "DbC: altitude must be non-negative");

    let t = T0 - LAPSE_RATE * altitude_m;
    let p = P0 * (t / T0).powf(GRAVITY * M_AIR / (R_GAS * LAPSE_RATE));
    let rho = p * M_AIR / (R_GAS * t);
    let mu = MU_REF * (t / T0).powf(MU_EXPONENT);

    let result = AtmosphereProperties {
        density: rho,
        viscosity: mu,
        temperature: t,
        pressure: p,
    };

    // DbC postcondition
    debug_assert!(result.density > 0.0, "DbC: density must be positive");
    debug_assert!(result.pressure > 0.0, "DbC: pressure must be positive");
    debug_assert!(
        result.temperature > 0.0,
        "DbC: temperature must be positive"
    );

    result
}

/// Convenience: get air density at a given altitude [kg/m³].
#[must_use]
pub fn air_density_at_altitude(altitude_m: f64) -> f64 {
    atmosphere_at_altitude(altitude_m).density
}

// ── Python bindings ──────────────────────────────────────────────────────────

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl AtmosphereProperties {
    /// Density [kg/m³].
    #[getter]
    fn density(&self) -> f64 {
        self.density
    }

    /// Viscosity [Pa·s].
    #[getter]
    fn viscosity(&self) -> f64 {
        self.viscosity
    }

    /// Temperature [K].
    #[getter]
    fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Pressure [Pa].
    #[getter]
    fn pressure(&self) -> f64 {
        self.pressure
    }

    /// Temperature in Celsius.
    #[getter]
    fn temperature_celsius(&self) -> f64 {
        self.temperature - 273.15
    }

    fn __repr__(&self) -> String {
        format!(
            "AtmosphereProperties(T={:.1}K, P={:.0}Pa, ρ={:.4}kg/m³)",
            self.temperature, self.pressure, self.density
        )
    }
}

/// WASM-exposed atmosphere calculation.
#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(js_name = "atmosphereAtAltitude")]
pub fn wasm_atmosphere_at_altitude(altitude_m: f64) -> AtmosphereProperties {
    atmosphere_at_altitude(altitude_m)
}

// ── Tests (TDD) ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Sea level properties match ISA standard values.
    #[test]
    fn test_sea_level() {
        let atm = atmosphere_at_altitude(0.0);
        assert!(
            (atm.temperature - 288.15).abs() < 0.01,
            "Sea-level temp should be ~288.15 K, got {}",
            atm.temperature
        );
        assert!(
            (atm.pressure - 101_325.0).abs() < 1.0,
            "Sea-level pressure should be ~101325 Pa, got {}",
            atm.pressure
        );
        assert!(
            (atm.density - 1.225).abs() < 0.01,
            "Sea-level density should be ~1.225 kg/m³, got {}",
            atm.density
        );
    }

    /// Test 2: Density decreases monotonically with altitude.
    #[test]
    fn test_density_decreases_with_altitude() {
        let sea = atmosphere_at_altitude(0.0);
        let mid = atmosphere_at_altitude(5000.0);
        let high = atmosphere_at_altitude(10000.0);
        assert!(mid.density < sea.density);
        assert!(high.density < mid.density);
    }

    /// Test 3: Temperature decreases at correct lapse rate.
    #[test]
    fn test_lapse_rate() {
        let low = atmosphere_at_altitude(0.0);
        let high = atmosphere_at_altitude(1000.0);
        let delta_t = low.temperature - high.temperature;
        assert!(
            (delta_t - 6.5).abs() < 0.01,
            "Lapse rate should drop 6.5K per 1000m, got {}",
            delta_t
        );
    }

    /// Test 4: Tropopause altitude (11km) gives valid properties.
    #[test]
    fn test_tropopause() {
        let atm = atmosphere_at_altitude(11000.0);
        assert!(atm.density > 0.0, "Density must be positive at 11km");
        assert!(
            atm.density < 0.5,
            "Density at 11km should be < 0.5 kg/m³, got {}",
            atm.density
        );
        assert!(
            atm.temperature > 200.0 && atm.temperature < 230.0,
            "Tropopause temp ~216.65 K, got {}",
            atm.temperature
        );
    }

    /// Test 5: Convenience function matches main function.
    #[test]
    fn test_density_convenience() {
        let rho = air_density_at_altitude(5000.0);
        let atm = atmosphere_at_altitude(5000.0);
        assert!((rho - atm.density).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "altitude must be non-negative")]
    fn test_negative_altitude_rejected() {
        let _ = atmosphere_at_altitude(-1.0);
    }

    #[test]
    #[should_panic(expected = "altitude must be finite")]
    fn test_non_finite_altitude_rejected() {
        let _ = atmosphere_at_altitude(f64::NAN);
    }

    /// Test 6: Viscosity increases with altitude (hotter → more viscous for gases... but
    /// ISA temp DECREASES, and our simplified model has viscosity ∝ T^0.76,
    /// so viscosity should DECREASE with altitude).
    #[test]
    fn test_viscosity_decreases_with_altitude() {
        let low = atmosphere_at_altitude(0.0);
        let high = atmosphere_at_altitude(10000.0);
        assert!(
            high.viscosity < low.viscosity,
            "Viscosity should decrease with altitude (lower T), got low={} high={}",
            low.viscosity,
            high.viscosity
        );
    }
}
