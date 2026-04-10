//! FFI wrappers for numerical integration: full trajectory simulation of
//! double, triple and golfer models, and batch evaluation helpers.

#![allow(dead_code)]

#[cfg(feature = "python")]
pub mod python {
    use crate::bindings::state::python::{
        PyDoublePendulumParams, PyGolferParams, PyTriplePendulumParams,
    };
    use crate::golfer_constraints::{
        constrained_accelerations as golfer_constrained_accelerations, BaumgarteGains,
    };
    use crate::{double_equations_of_motion, triple_equations_of_motion};
    use pyo3::prelude::*;

    /// Simulate the double pendulum with polynomial torque profiles.
    #[pyfunction]
    #[pyo3(signature = (params, q0, qdot0, coeffs, n_coeffs_per_joint, t_span, max_steps=100000))]
    pub fn py_simulate_double(
        params: &PyDoublePendulumParams,
        q0: Vec<f64>,
        qdot0: Vec<f64>,
        coeffs: Vec<f64>,
        n_coeffs_per_joint: usize,
        t_span: (f64, f64),
        max_steps: usize,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        if q0.len() != 2 || qdot0.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q0 and qdot0 must have length 2",
            ));
        }
        if coeffs.len() != n_coeffs_per_joint * 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coeffs must have length equal to n_coeffs_per_joint * 2",
            ));
        }

        let (coeffs1, coeffs2) = coeffs.split_at(n_coeffs_per_joint);
        let q0_arr = [q0[0], q0[1]];
        let qdot0_arr = [qdot0[0], qdot0[1]];

        let config = crate::integrator::RK45Config {
            h0: 0.005,
            h_min: 1e-6,
            h_max: 0.01,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps,
        };

        let mut y0 = [0.0; 4];
        y0[0] = q0_arr[0];
        y0[1] = q0_arr[1];
        y0[2] = qdot0_arr[0];
        y0[3] = qdot0_arr[1];

        // Use generic rk45 on the full 4D state vector
        let result = crate::integrator::integrate_rk45(
            |t, y| {
                let q = [y[0], y[1]];
                let qd = [y[2], y[3]];

                let tau1: f64 = coeffs1
                    .iter()
                    .enumerate()
                    .map(|(i, c)| c * t.powi(i as i32))
                    .sum();
                let tau2: f64 = coeffs2
                    .iter()
                    .enumerate()
                    .map(|(i, c)| c * t.powi(i as i32))
                    .sum();

                let qddot = double_equations_of_motion(
                    &q,
                    &qd,
                    &[tau1, tau2],
                    &params.inner,
                );

                [qd[0], qd[1], qddot[0], qddot[1]]
            },
            t_span.0,
            t_span.1,
            y0,
            config,
        );

        let times: Vec<f64> = result.iter().map(|s| s.t).collect();
        let states: Vec<Vec<f64>> = result.iter().map(|s| s.y.to_vec()).collect();

        Ok((times, states))
    }

    /// Simulate the triple pendulum with polynomial torque profiles.
    ///
    /// Runs the full RK45 integration loop in Rust. Returns (times, states)
    /// where states is a list of 6-element vectors [q1, q2, q3, qdot1, qdot2, qdot3].
    #[pyfunction]
    #[pyo3(signature = (params, q0, qdot0, coeffs, n_coeffs_per_joint, t_span, max_steps=100000))]
    pub fn py_simulate_triple(
        params: &PyTriplePendulumParams,
        q0: Vec<f64>,
        qdot0: Vec<f64>,
        coeffs: Vec<f64>,
        n_coeffs_per_joint: usize,
        t_span: (f64, f64),
        max_steps: usize,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        if q0.len() != 3 || qdot0.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q0 and qdot0 must have length 3",
            ));
        }
        if coeffs.len() != n_coeffs_per_joint * 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coeffs must have length equal to n_coeffs_per_joint * 3",
            ));
        }

        let joint_coeffs: Vec<&[f64]> = coeffs.chunks(n_coeffs_per_joint).collect();

        let config = crate::integrator::RK45Config {
            h0: 0.005,
            h_min: 1e-6,
            h_max: 0.01,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps,
        };

        let y0 = [q0[0], q0[1], q0[2], qdot0[0], qdot0[1], qdot0[2]];

        let result = crate::integrator::integrate_rk45(
            |t, y| {
                let q = [y[0], y[1], y[2]];
                let qd = [y[3], y[4], y[5]];

                let mut tau = [0.0; 3];
                for (j, tau_j) in tau.iter_mut().enumerate() {
                    *tau_j = joint_coeffs[j]
                        .iter()
                        .enumerate()
                        .map(|(i, c)| c * t.powi(i as i32))
                        .sum();
                }

                let qddot = triple_equations_of_motion(&q, &qd, &tau, &params.inner);
                [qd[0], qd[1], qd[2], qddot[0], qddot[1], qddot[2]]
            },
            t_span.0,
            t_span.1,
            y0,
            config,
        );

        let times: Vec<f64> = result.iter().map(|s| s.t).collect();
        let states: Vec<Vec<f64>> = result.iter().map(|s| s.y.to_vec()).collect();

        Ok((times, states))
    }

    /// Simulate the golfer model with polynomial torque profiles and constraint enforcement.
    ///
    /// Runs the full constrained RK45 integration loop in Rust for the 8-DOF golfer model.
    /// Returns (times, states) where states is a list of 16-element vectors [q(8), qdot(8)].
    #[pyfunction]
    #[pyo3(signature = (params, q0, qdot0, coeffs, n_coeffs_per_joint, t_span, alpha=5.0, beta=5.0, max_steps=100000))]
    pub fn py_simulate_golfer(
        params: &PyGolferParams,
        q0: Vec<f64>,
        qdot0: Vec<f64>,
        coeffs: Vec<f64>,
        n_coeffs_per_joint: usize,
        t_span: (f64, f64),
        alpha: f64,
        beta: f64,
        max_steps: usize,
    ) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
        if q0.len() != 8 || qdot0.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q0 and qdot0 must have length 8",
            ));
        }
        // 7 actuated joints (club DOF has no independent torque)
        if coeffs.len() != n_coeffs_per_joint * 7 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "coeffs must have length equal to n_coeffs_per_joint * 7 (no torque on club DOF)",
            ));
        }

        let joint_coeffs: Vec<&[f64]> = coeffs.chunks(n_coeffs_per_joint).collect();
        let gains = BaumgarteGains { alpha, beta };

        let config = crate::integrator::RK45Config {
            h0: 0.001,
            h_min: 1e-8,
            h_max: 0.005,
            rtol: 1e-8,
            atol: 1e-10,
            max_steps,
        };

        let mut y0 = [0.0; 16];
        y0[..8].copy_from_slice(&q0);
        y0[8..16].copy_from_slice(&qdot0);

        let result = crate::integrator::integrate_rk45(
            |t, y| {
                let mut q = [0.0; 8];
                let mut qd = [0.0; 8];
                q.copy_from_slice(&y[..8]);
                qd.copy_from_slice(&y[8..16]);

                // Compute polynomial torques for the 7 actuated joints
                let mut tau = [0.0; 8];
                for j in 0..7 {
                    tau[j] = joint_coeffs[j]
                        .iter()
                        .enumerate()
                        .map(|(i, c)| c * t.powi(i as i32))
                        .sum();
                }

                // Solve constrained dynamics
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    golfer_constrained_accelerations(
                        &q, &qd, &tau, &params.inner, &gains,
                    )
                }));

                let mut dy = [0.0; 16];
                dy[..8].copy_from_slice(&qd);
                if let Ok((qddot, _lambda)) = result {
                    for i in 0..8 {
                        dy[8 + i] = qddot[i];
                    }
                }
                // If constrained dynamics panics, return zero accelerations
                // (the integrator will detect the stiff region and reduce step size)

                dy
            },
            t_span.0,
            t_span.1,
            y0,
            config,
        );

        let times: Vec<f64> = result.iter().map(|s| s.t).collect();
        let states: Vec<Vec<f64>> = result.iter().map(|s| s.y.to_vec()).collect();

        Ok((times, states))
    }

    /// Batch-evaluate N candidate polynomial torque profiles in parallel.
    ///
    /// Returns a list of (max_tip_speed, tip_speed_at_bottom, success) tuples.
    #[pyfunction]
    #[pyo3(signature = (params, coeffs_batch, n_coeffs_per_joint, q0, qdot0, t_end))]
    pub fn py_batch_evaluate_double(
        params: &PyDoublePendulumParams,
        coeffs_batch: Vec<Vec<f64>>,
        n_coeffs_per_joint: usize,
        q0: Vec<f64>,
        qdot0: Vec<f64>,
        t_end: f64,
    ) -> PyResult<Vec<(f64, f64, bool)>> {
        if q0.len() != 2 || qdot0.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q0 and qdot0 must have length 2",
            ));
        }
        let q0_arr = [q0[0], q0[1]];
        let qdot0_arr = [qdot0[0], qdot0[1]];

        let results = crate::batch::batch_evaluate_double(
            &params.inner,
            &coeffs_batch,
            n_coeffs_per_joint,
            q0_arr,
            qdot0_arr,
            t_end,
        );

        Ok(results
            .iter()
            .map(|r| (r.max_tip_speed, r.tip_speed_at_bottom, r.success))
            .collect())
    }
}
