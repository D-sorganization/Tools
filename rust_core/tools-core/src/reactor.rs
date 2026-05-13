//! Numerical integration for 1D Plug Flow Reactors (TRC).
//!
//! Provides a standard 4th-order Runge-Kutta (RK4) integrator for generic ODE systems.

pub trait OdeSystem {
    /// Dimension of the state vector
    fn state_dim(&self) -> usize;

    /// Computes the derivatives (dy/dz) given independent variable z and state y.
    /// `dy` must be mutated in place.
    fn derivatives(&self, z: f64, y: &[f64], dy: &mut [f64]);
}

/// A 4th-order Runge-Kutta integrator for 1D ODE systems.
pub struct Rk4Integrator {
    pub step_size: f64,
}

impl Rk4Integrator {
    pub fn new(step_size: f64) -> Self {
        assert!(step_size > 0.0, "step size must be positive");
        Self { step_size }
    }

    /// Integrates the ODE system from z_start to z_end with initial state y0.
    /// Returns a vector of tuples `(z, y)` representing the solution trajectory.
    pub fn integrate<S: OdeSystem>(
        &self,
        system: &S,
        z_start: f64,
        z_end: f64,
        y0: &[f64],
    ) -> Vec<(f64, Vec<f64>)> {
        let dim = system.state_dim();
        assert_eq!(y0.len(), dim, "Initial state length must match system dimension");

        let mut trajectory = Vec::new();
        let mut z = z_start;
        let mut y = y0.to_vec();

        let mut k1 = vec![0.0; dim];
        let mut k2 = vec![0.0; dim];
        let mut k3 = vec![0.0; dim];
        let mut k4 = vec![0.0; dim];
        let mut y_temp = vec![0.0; dim];

        trajectory.push((z, y.clone()));

        while z < z_end {
            let h = if z + self.step_size > z_end {
                z_end - z
            } else {
                self.step_size
            };

            // k1
            system.derivatives(z, &y, &mut k1);

            // k2
            for i in 0..dim {
                y_temp[i] = y[i] + 0.5 * h * k1[i];
            }
            system.derivatives(z + 0.5 * h, &y_temp, &mut k2);

            // k3
            for i in 0..dim {
                y_temp[i] = y[i] + 0.5 * h * k2[i];
            }
            system.derivatives(z + 0.5 * h, &y_temp, &mut k3);

            // k4
            for i in 0..dim {
                y_temp[i] = y[i] + h * k3[i];
            }
            system.derivatives(z + h, &y_temp, &mut k4);

            // update
            for i in 0..dim {
                y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
            z += h;

            trajectory.push((z, y.clone()));
        }

        trajectory
    }
}

/// A simplified 1D Tubular Reactor ODE system for TRC kinetics
pub struct Trc1DSystem {
    pub pre_exponential: f64,
    pub activation_energy: f64, // J/mol
    pub heat_of_reaction: f64,  // J/mol
    pub density: f64,           // kg/m3
    pub cp: f64,                // J/(kg.K)
    pub velocity: f64,          // m/s
}

impl OdeSystem for Trc1DSystem {
    fn state_dim(&self) -> usize {
        2 // state: [conversion, temperature]
    }

    fn derivatives(&self, _z: f64, y: &[f64], dy: &mut [f64]) {
        let x = y[0];
        let t = y[1];

        // Arrhenius rate: k = A * exp(-E / RT)
        let rate = self.pre_exponential
            * (-self.activation_energy / (crate::engineering::R_UNIVERSAL * t)).exp();

        // simple first order kinetics: dX/dz = (rate * (1 - X)) / velocity
        let dx_dz = rate * (1.0 - x) / self.velocity;

        // dT/dz = (-heat_of_reaction * rate * (1 - X)) / (density * Cp * velocity)
        let dt_dz =
            (-self.heat_of_reaction * rate * (1.0 - x)) / (self.density * self.cp * self.velocity);

        dy[0] = dx_dz;
        dy[1] = dt_dz;
    }
}

#[cfg(feature = "python")]
pub mod py_bindings {
    use super::*;
    use pyo3::prelude::*;

    #[pyclass]
    pub struct PyTrc1DSystem {
        inner: Trc1DSystem,
    }

    #[pymethods]
    impl PyTrc1DSystem {
        #[new]
        pub fn new(
            pre_exponential: f64,
            activation_energy: f64,
            heat_of_reaction: f64,
            density: f64,
            cp: f64,
            velocity: f64,
        ) -> Self {
            PyTrc1DSystem {
                inner: Trc1DSystem {
                    pre_exponential,
                    activation_energy,
                    heat_of_reaction,
                    density,
                    cp,
                    velocity,
                },
            }
        }

        /// Integrates the TRC ODE system from z_start to z_end with given step size and initial state [conversion, temperature].
        /// Returns a list of tuples (z, conversion, temperature).
        pub fn integrate(&self, step_size: f64, z_start: f64, z_end: f64, y0: Vec<f64>) -> Vec<(f64, f64, f64)> {
            let integrator = Rk4Integrator::new(step_size);
            let trajectory = integrator.integrate(&self.inner, z_start, z_end, &y0);
            
            trajectory
                .into_iter()
                .map(|(z, y)| (z, y[0], y[1]))
                .collect()
        }
    }
}
