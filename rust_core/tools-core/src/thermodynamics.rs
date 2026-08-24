//! Thermodynamics — NASA-7 polynomial evaluation and thermochemical properties.

/// NASA-7 Polynomial coefficients for a specific temperature range.
#[derive(Debug, Clone, Copy)]
pub struct Nasa7Coefficients {
    pub a1: f64,
    pub a2: f64,
    pub a3: f64,
    pub a4: f64,
    pub a5: f64,
    pub a6: f64,
    pub a7: f64,
}

impl Nasa7Coefficients {
    /// Computes dimensionless specific heat capacity (Cp/R)
    pub fn cp_over_r(&self, t: f64) -> f64 {
        self.a1 + self.a2 * t + self.a3 * t.powi(2) + self.a4 * t.powi(3) + self.a5 * t.powi(4)
    }

    /// Computes dimensionless enthalpy (H/RT)
    pub fn h_over_rt(&self, t: f64) -> f64 {
        self.a1
            + self.a2 * t / 2.0
            + self.a3 * t.powi(2) / 3.0
            + self.a4 * t.powi(3) / 4.0
            + self.a5 * t.powi(4) / 5.0
            + self.a6 / t
    }

    /// Computes dimensionless entropy (S/R)
    pub fn s_over_r(&self, t: f64) -> f64 {
        self.a1 * t.ln()
            + self.a2 * t
            + self.a3 * t.powi(2) / 2.0
            + self.a4 * t.powi(3) / 3.0
            + self.a5 * t.powi(4) / 4.0
            + self.a7
    }
}

/// A species represented by two sets of NASA-7 polynomials (low and high temp).
#[derive(Debug, Clone)]
pub struct Nasa7Species {
    pub name: String,
    pub molar_mass: f64, // kg/mol
    pub t_mid: f64,
    pub low_temp: Nasa7Coefficients,
    pub high_temp: Nasa7Coefficients,
}

impl Nasa7Species {
    fn get_coeffs(&self, t: f64) -> &Nasa7Coefficients {
        if t < self.t_mid {
            &self.low_temp
        } else {
            &self.high_temp
        }
    }

    /// Specific heat capacity Cp in J/(mol·K)
    pub fn cp(&self, t: f64) -> f64 {
        self.get_coeffs(t).cp_over_r(t) * crate::engineering::R_UNIVERSAL
    }

    /// Enthalpy H in J/mol
    pub fn enthalpy(&self, t: f64) -> f64 {
        self.get_coeffs(t).h_over_rt(t) * crate::engineering::R_UNIVERSAL * t
    }

    /// Entropy S in J/(mol·K)
    pub fn entropy(&self, t: f64) -> f64 {
        self.get_coeffs(t).s_over_r(t) * crate::engineering::R_UNIVERSAL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engineering::R_UNIVERSAL;
    use proptest::prelude::*;

    /// NASA-7 coefficients for N2 (low-temp range, GRI-Mech 3.0), a well-known
    /// reference species. Cp/R at 300 K ≈ 3.5 (diatomic, ~7/2 R).
    fn n2_low() -> Nasa7Coefficients {
        Nasa7Coefficients {
            a1: 3.298_677_0,
            a2: 1.408_240_4e-3,
            a3: -3.963_222_0e-6,
            a4: 5.641_515_0e-9,
            a5: -2.444_854_0e-12,
            a6: -1.020_899_9e3,
            a7: 3.950_372_0,
        }
    }

    fn n2_high() -> Nasa7Coefficients {
        Nasa7Coefficients {
            a1: 2.926_640_0,
            a2: 1.487_976_8e-3,
            a3: -5.684_760_0e-7,
            a4: 1.009_703_8e-10,
            a5: -6.753_351_0e-15,
            a6: -9.227_977_0e2,
            a7: 5.980_528_0,
        }
    }

    fn n2_species() -> Nasa7Species {
        Nasa7Species {
            name: "N2".to_string(),
            molar_mass: 0.028_014,
            t_mid: 1000.0,
            low_temp: n2_low(),
            high_temp: n2_high(),
        }
    }

    #[test]
    fn cp_over_r_diatomic_near_seven_halves() {
        // A diatomic ideal gas has Cp/R ≈ 7/2 = 3.5 at moderate temperature.
        let cp_r = n2_low().cp_over_r(300.0);
        assert!(
            (cp_r - 3.5).abs() < 0.05,
            "Cp/R for N2 at 300 K should be ~3.5, got {cp_r}"
        );
    }

    #[test]
    fn cp_is_cp_over_r_times_r() {
        // `cp` must equal `cp_over_r * R_UNIVERSAL` exactly (DRY wiring check).
        let s = n2_species();
        let t = 500.0;
        assert!((s.cp(t) - s.low_temp.cp_over_r(t) * R_UNIVERSAL).abs() < 1e-9);
    }

    #[test]
    fn get_coeffs_switches_at_t_mid() {
        // Below t_mid uses low_temp; at/above uses high_temp. We can observe the
        // switch because the two coefficient sets differ at the boundary.
        let s = n2_species();
        let just_below = s.cp(999.9);
        let at_mid = s.cp(1000.0);
        // The NASA-7 fit is continuous-ish but the two polynomials are distinct;
        // the values straddle the boundary and must both be finite & positive.
        assert!(just_below > 0.0 && at_mid > 0.0);
        // Selecting high-temp coeffs at exactly t_mid must match high_temp eval.
        assert!((at_mid - s.high_temp.cp_over_r(1000.0) * R_UNIVERSAL).abs() < 1e-9);
    }

    #[test]
    fn enthalpy_and_entropy_finite_and_signed() {
        let s = n2_species();
        let h = s.enthalpy(800.0);
        let entropy = s.entropy(800.0);
        assert!(h.is_finite());
        // Standard molar entropy of a gas is positive.
        assert!(entropy > 0.0, "entropy should be positive, got {entropy}");
    }

    #[test]
    fn enthalpy_increases_with_temperature() {
        // dH/dT = Cp > 0, so enthalpy is monotonically increasing in T.
        let s = n2_species();
        let h_low = s.enthalpy(400.0);
        let h_high = s.enthalpy(600.0);
        assert!(
            h_high > h_low,
            "enthalpy should increase with T: {h_low} -> {h_high}"
        );
    }

    proptest! {
        /// Invariant: over a physically reasonable temperature band the NASA-7
        /// Cp/R for N2 stays in a sane range (diatomic gas: 3 < Cp/R < 6) and
        /// `cp()` is always exactly `R_UNIVERSAL` times the dimensionless form.
        #[test]
        fn prop_cp_consistency_and_bounds(t in 250.0_f64..3000.0_f64) {
            let s = n2_species();
            let cp = s.cp(t);
            let coeffs = if t < s.t_mid { &s.low_temp } else { &s.high_temp };
            prop_assert!((cp - coeffs.cp_over_r(t) * R_UNIVERSAL).abs() < 1e-6);
            let cp_r = cp / R_UNIVERSAL;
            prop_assert!((3.0..6.0).contains(&cp_r), "Cp/R out of range: {cp_r} at T={t}");
        }
    }
}

#[cfg(feature = "python")]
pub mod py_bindings {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass(from_py_object)]
    #[derive(Clone)]
    pub struct PyNasa7Species {
        inner: Nasa7Species,
    }

    #[pymethods]
    impl PyNasa7Species {
        #[new]
        #[pyo3(signature = (name, molar_mass, t_mid, low_coeffs, high_coeffs))]
        pub fn new(
            name: String,
            molar_mass: f64,
            t_mid: f64,
            low_coeffs: [f64; 7],
            high_coeffs: [f64; 7],
        ) -> Self {
            PyNasa7Species {
                inner: Nasa7Species {
                    name,
                    molar_mass,
                    t_mid,
                    low_temp: Nasa7Coefficients {
                        a1: low_coeffs[0],
                        a2: low_coeffs[1],
                        a3: low_coeffs[2],
                        a4: low_coeffs[3],
                        a5: low_coeffs[4],
                        a6: low_coeffs[5],
                        a7: low_coeffs[6],
                    },
                    high_temp: Nasa7Coefficients {
                        a1: high_coeffs[0],
                        a2: high_coeffs[1],
                        a3: high_coeffs[2],
                        a4: high_coeffs[3],
                        a5: high_coeffs[4],
                        a6: high_coeffs[5],
                        a7: high_coeffs[6],
                    },
                },
            }
        }

        pub fn cp(&self, t: f64) -> f64 {
            self.inner.cp(t)
        }

        pub fn enthalpy(&self, t: f64) -> f64 {
            self.inner.enthalpy(t)
        }

        pub fn entropy(&self, t: f64) -> f64 {
            self.inner.entropy(t)
        }
    }
}
