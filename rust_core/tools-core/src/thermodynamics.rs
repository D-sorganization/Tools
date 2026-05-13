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

#[cfg(feature = "python")]
pub mod py_bindings {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass]
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
