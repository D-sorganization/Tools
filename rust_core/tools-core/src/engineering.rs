//! Engineering calculation primitives — fluid mechanics, thermodynamics, unit conversion.
//!
//! This module provides canonical implementations of engineering formulas
//! used by multiple calculator applications across the fleet.
//!
//! # Design by Contract
//! - All physical quantities must be in SI units unless otherwise stated
//! - All pressures are absolute (Pa)
//! - All temperatures are in Kelvin

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Universal gas constant (J/(mol·K))
pub const R_UNIVERSAL: f64 = 8.314_462_618_153_24;

/// Stefan-Boltzmann constant (W/(m²·K⁴))
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Standard atmospheric pressure (Pa)
pub const ATM_PA: f64 = 101_325.0;

/// Standard gravity (m/s²)
pub const G_STANDARD: f64 = 9.806_65;

/// Absolute zero offset (°C to K)
pub const CELSIUS_TO_KELVIN: f64 = 273.15;

// ---------------------------------------------------------------------------
// Unit conversions
// ---------------------------------------------------------------------------

/// Convert temperature from Celsius to Kelvin.
pub fn celsius_to_kelvin(celsius: f64) -> f64 {
    celsius + CELSIUS_TO_KELVIN
}

/// Convert temperature from Kelvin to Celsius.
pub fn kelvin_to_celsius(kelvin: f64) -> f64 {
    kelvin - CELSIUS_TO_KELVIN
}

/// Convert temperature from Fahrenheit to Kelvin.
pub fn fahrenheit_to_kelvin(fahrenheit: f64) -> f64 {
    (fahrenheit - 32.0) * 5.0 / 9.0 + CELSIUS_TO_KELVIN
}

/// Convert temperature from Kelvin to Fahrenheit.
pub fn kelvin_to_fahrenheit(kelvin: f64) -> f64 {
    (kelvin - CELSIUS_TO_KELVIN) * 9.0 / 5.0 + 32.0
}

/// Convert pressure from bar to Pa.
pub fn bar_to_pa(bar: f64) -> f64 {
    bar * 1e5
}

/// Convert pressure from psi to Pa.
pub fn psi_to_pa(psi: f64) -> f64 {
    psi * 6_894.757_293_168
}

/// Convert pressure from Pa to psi.
pub fn pa_to_psi(pa: f64) -> f64 {
    pa / 6_894.757_293_168
}

// ---------------------------------------------------------------------------
// Fluid mechanics
// ---------------------------------------------------------------------------

/// Compute the Reynolds number for internal pipe flow.
///
/// # Arguments
/// - `velocity` — flow velocity (m/s)
/// - `diameter` — pipe inner diameter (m)
/// - `density` — fluid density (kg/m³)
/// - `viscosity` — dynamic viscosity (Pa·s)
///
/// # Preconditions
/// - `viscosity > 0`
/// - `diameter > 0`
pub fn reynolds_number(velocity: f64, diameter: f64, density: f64, viscosity: f64) -> f64 {
    debug_assert!(viscosity > 0.0, "viscosity must be positive");
    debug_assert!(diameter > 0.0, "diameter must be positive");
    density * velocity * diameter / viscosity
}

/// Compute friction factor using Churchill (1977) correlation.
/// Valid for all flow regimes (laminar + turbulent + transition).
///
/// # Arguments
/// - `re` — Reynolds number (dimensionless)
/// - `roughness` — pipe roughness (m)
/// - `diameter` — pipe diameter (m)
///
/// # Returns
/// Darcy friction factor (dimensionless)
pub fn churchill_friction_factor(re: f64, roughness: f64, diameter: f64) -> f64 {
    debug_assert!(re > 0.0, "Reynolds number must be positive");
    debug_assert!(diameter > 0.0, "diameter must be positive");

    let a = (2.457 * ((7.0 / re).powf(0.9) + 0.27 * roughness / diameter).ln()).powf(16.0);
    let b = (37530.0 / re).powf(16.0);
    let term = 8.0 / re;
    8.0 * (term.powf(12.0) + 1.0 / (a + b).powf(1.5)).powf(1.0 / 12.0)
}

/// Compute pressure drop using Darcy-Weisbach equation.
///
/// ΔP = f · (L/D) · (ρ·v²/2)
///
/// # Arguments
/// - `friction_factor` — Darcy friction factor (dimensionless)
/// - `length` — pipe length (m)
/// - `diameter` — pipe diameter (m)
/// - `density` — fluid density (kg/m³)
/// - `velocity` — flow velocity (m/s)
///
/// # Returns
/// Pressure drop in Pa
pub fn darcy_weisbach_pressure_drop(
    friction_factor: f64,
    length: f64,
    diameter: f64,
    density: f64,
    velocity: f64,
) -> f64 {
    debug_assert!(diameter > 0.0, "diameter must be positive");
    friction_factor * (length / diameter) * (density * velocity * velocity / 2.0)
}

/// Compute volumetric flow rate from velocity and pipe area.
///
/// Q = v · π · D²/4
pub fn flow_rate_from_velocity(velocity: f64, diameter: f64) -> f64 {
    debug_assert!(diameter > 0.0, "diameter must be positive");
    velocity * PI * diameter * diameter / 4.0
}

/// Convert volumetric flow rate to mass flow rate.
pub fn volumetric_to_mass_flow(volumetric_flow: f64, density: f64) -> f64 {
    volumetric_flow * density
}

// ---------------------------------------------------------------------------
// Thermodynamics
// ---------------------------------------------------------------------------

/// Compute ideal gas density: ρ = P·M / (R·T)
///
/// # Arguments
/// - `pressure` — absolute pressure (Pa)
/// - `molar_mass` — molar mass (kg/mol)
/// - `temperature` — temperature (K)
pub fn ideal_gas_density(pressure: f64, molar_mass: f64, temperature: f64) -> f64 {
    debug_assert!(temperature > 0.0, "temperature must be positive");
    pressure * molar_mass / (R_UNIVERSAL * temperature)
}

/// Compute compressibility factor Z for real gas (simple van der Waals approximation).
///
/// Z = PV/(nRT) ≈ 1 + B'P where B' = b - a/(RT)
///
/// # Arguments
/// - `pressure` — absolute pressure (Pa)
/// - `temperature` — temperature (K)
/// - `a_vdw` — van der Waals `a` parameter (Pa·m⁶/mol²)
/// - `b_vdw` — van der Waals `b` parameter (m³/mol)
pub fn compressibility_factor_vdw(pressure: f64, temperature: f64, a_vdw: f64, b_vdw: f64) -> f64 {
    debug_assert!(temperature > 0.0, "temperature must be positive");
    let b_prime = b_vdw - a_vdw / (R_UNIVERSAL * temperature);
    1.0 + b_prime * pressure / (R_UNIVERSAL * temperature)
}

/// Compute isentropic compressor/expander work per unit mass.
///
/// w = (k/(k-1)) · R/M · T1 · [(P2/P1)^((k-1)/k) - 1]
///
/// # Arguments
/// - `t1` — inlet temperature (K)
/// - `p1` — inlet pressure (Pa)
/// - `p2` — outlet pressure (Pa)
/// - `k` — ratio of specific heats (Cp/Cv)
/// - `molar_mass` — molar mass (kg/mol)
pub fn isentropic_work(t1: f64, p1: f64, p2: f64, k: f64, molar_mass: f64) -> f64 {
    debug_assert!(t1 > 0.0, "inlet temperature must be positive");
    debug_assert!(p1 > 0.0, "inlet pressure must be positive");
    debug_assert!(p2 > 0.0, "outlet pressure must be positive");
    debug_assert!(k > 1.0, "heat capacity ratio must be > 1");
    debug_assert!(molar_mass > 0.0, "molar mass must be positive");

    let exponent = (k - 1.0) / k;
    let r_specific = R_UNIVERSAL / molar_mass;
    (k / (k - 1.0)) * r_specific * t1 * ((p2 / p1).powf(exponent) - 1.0)
}

// ---------------------------------------------------------------------------
// Heat transfer
// ---------------------------------------------------------------------------

/// Compute convective heat transfer rate: Q = h·A·ΔT
pub fn convective_heat_transfer(h: f64, area: f64, delta_t: f64) -> f64 {
    h * area * delta_t
}

/// Compute radiative heat transfer rate: Q = ε·σ·A·(T1⁴ - T2⁴)
pub fn radiative_heat_transfer(emissivity: f64, area: f64, t1: f64, t2: f64) -> f64 {
    debug_assert!(
        (0.0..=1.0).contains(&emissivity),
        "emissivity must be in [0, 1]"
    );
    emissivity * STEFAN_BOLTZMANN * area * (t1.powi(4) - t2.powi(4))
}

/// Log-mean temperature difference for a heat exchanger.
///
/// LMTD = (ΔT1 - ΔT2) / ln(ΔT1/ΔT2)
pub fn lmtd(delta_t1: f64, delta_t2: f64) -> f64 {
    debug_assert!(delta_t1 > 0.0, "delta_t1 must be positive");
    debug_assert!(delta_t2 > 0.0, "delta_t2 must be positive");
    if (delta_t1 - delta_t2).abs() < 1e-10 {
        // Avoid division by zero when ΔT1 ≈ ΔT2
        return delta_t1;
    }
    (delta_t1 - delta_t2) / (delta_t1 / delta_t2).ln()
}

// ---------------------------------------------------------------------------
// PyO3 bindings (feature-gated)
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub mod py_bindings {
    use super::*;
    use pyo3::prelude::*;

    #[pyfunction]
    #[pyo3(name = "reynolds_number")]
    pub fn py_reynolds_number(velocity: f64, diameter: f64, density: f64, viscosity: f64) -> f64 {
        reynolds_number(velocity, diameter, density, viscosity)
    }

    #[pyfunction]
    #[pyo3(name = "churchill_friction_factor")]
    pub fn py_churchill_friction_factor(re: f64, roughness: f64, diameter: f64) -> f64 {
        churchill_friction_factor(re, roughness, diameter)
    }

    #[pyfunction]
    #[pyo3(name = "darcy_weisbach_pressure_drop")]
    pub fn py_darcy_weisbach(
        friction_factor: f64,
        length: f64,
        diameter: f64,
        density: f64,
        velocity: f64,
    ) -> f64 {
        darcy_weisbach_pressure_drop(friction_factor, length, diameter, density, velocity)
    }

    #[pyfunction]
    #[pyo3(name = "ideal_gas_density")]
    pub fn py_ideal_gas_density(pressure: f64, molar_mass: f64, temperature: f64) -> f64 {
        ideal_gas_density(pressure, molar_mass, temperature)
    }

    #[pyfunction]
    #[pyo3(name = "isentropic_work")]
    pub fn py_isentropic_work(t1: f64, p1: f64, p2: f64, k: f64, molar_mass: f64) -> f64 {
        isentropic_work(t1, p1, p2, k, molar_mass)
    }

    #[pyfunction]
    #[pyo3(name = "lmtd")]
    pub fn py_lmtd(delta_t1: f64, delta_t2: f64) -> f64 {
        lmtd(delta_t1, delta_t2)
    }

    #[pyfunction]
    #[pyo3(name = "celsius_to_kelvin")]
    pub fn py_celsius_to_kelvin(celsius: f64) -> f64 {
        celsius_to_kelvin(celsius)
    }

    #[pyfunction]
    #[pyo3(name = "kelvin_to_celsius")]
    pub fn py_kelvin_to_celsius(kelvin: f64) -> f64 {
        kelvin_to_celsius(kelvin)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celsius_kelvin_roundtrip() {
        let c = 100.0;
        let k = celsius_to_kelvin(c);
        assert!((k - 373.15).abs() < 1e-10);
        assert!((kelvin_to_celsius(k) - c).abs() < 1e-10);
    }

    #[test]
    fn test_fahrenheit_kelvin_boiling() {
        let f = 212.0; // boiling point of water
        let k = fahrenheit_to_kelvin(f);
        assert!((k - 373.15).abs() < 0.01);
    }

    #[test]
    fn test_pressure_conversions() {
        assert!((bar_to_pa(1.0) - 1e5).abs() < 1e-5);
        let psi_val = 14.696; // ~1 atm
        let pa_val = psi_to_pa(psi_val);
        assert!((pa_val - ATM_PA).abs() < 100.0); // within 100 Pa
    }

    #[test]
    fn test_reynolds_laminar() {
        // Water at ~1 m/s in 25mm pipe: Re ≈ 25000
        let re = reynolds_number(1.0, 0.025, 1000.0, 0.001);
        assert!((re - 25000.0).abs() < 0.1);
    }

    #[test]
    fn test_flow_rate() {
        let q = flow_rate_from_velocity(1.0, 0.1); // 100mm pipe, 1 m/s
        let expected = PI * 0.01 / 4.0; // π·D²/4
        assert!((q - expected).abs() < 1e-10);
    }

    #[test]
    fn test_darcy_weisbach() {
        // Simple check: known f, L, D, rho, v
        let dp = darcy_weisbach_pressure_drop(0.02, 100.0, 0.1, 1000.0, 2.0);
        // dp = 0.02 * (100/0.1) * (1000 * 4 / 2) = 0.02 * 1000 * 2000 = 40000 Pa
        assert!((dp - 40000.0).abs() < 0.1);
    }

    #[test]
    fn test_ideal_gas_density_air() {
        // Air at STP: ρ ≈ 1.225 kg/m³
        let rho = ideal_gas_density(ATM_PA, 0.029, 288.15);
        assert!((rho - 1.225).abs() < 0.02);
    }

    #[test]
    fn test_isentropic_compression() {
        // Compress air (k=1.4, M=0.029) from 1 atm to 2 atm at 300K
        let w = isentropic_work(300.0, ATM_PA, 2.0 * ATM_PA, 1.4, 0.029);
        // w should be positive (work input needed)
        assert!(w > 0.0, "compression work should be positive");
        // Rough check: ~ 60-70 kJ/kg
        assert!(w > 50000.0 && w < 80000.0, "w = {} J/kg", w);
    }

    #[test]
    fn test_lmtd_symmetric() {
        // When ΔT1 = ΔT2, LMTD = ΔT
        let result = lmtd(50.0, 50.0);
        assert!((result - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_lmtd_asymmetric() {
        let result = lmtd(100.0, 50.0);
        // LMTD = (100-50) / ln(100/50) = 50 / ln(2) ≈ 72.13
        assert!((result - 72.13).abs() < 0.1);
    }

    #[test]
    fn test_radiative_heat_transfer() {
        // Black body (ε=1), 1 m², from 500K to 300K
        let q = radiative_heat_transfer(1.0, 1.0, 500.0, 300.0);
        // Q = σ(500⁴ - 300⁴) = 5.67e-8 * (6.25e10 - 8.1e9) ≈ 3.1 kW
        assert!(q > 2500.0 && q < 3500.0, "Q = {} W", q);
    }

    #[test]
    fn test_convective_heat_transfer() {
        let q = convective_heat_transfer(50.0, 2.0, 30.0);
        assert!((q - 3000.0).abs() < 1e-10);
    }
}
