//! FFI wrappers for constraint forces, Jacobians, mobility/force ellipsoids,
//! and CMA-ES optimisation of torque coefficients.

#![allow(dead_code)]

#[cfg(feature = "python")]
pub mod python {
    use crate::bindings::state::python::{PyCmaEsResult, PyGolferParams};
    use crate::bindings::state::to_array_8;
    use crate::golfer_constraints::{
        constrained_accelerations as golfer_constrained_accelerations,
        project_to_constraints as golfer_project_to_constraints,
        project_velocity as golfer_project_velocity, BaumgarteGains,
    };
    use crate::{constraint_jacobian, constraint_vector};
    use pyo3::prelude::*;
    use std::collections::HashMap;

    /// Golfer constraint vector
    #[pyfunction]
    pub fn py_golfer_constraint_vector(
        q: Vec<f64>,
        params: &PyGolferParams,
    ) -> PyResult<Vec<f64>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 8",
            ));
        }
        let mut q_arr = [0.0; 8];
        q_arr.copy_from_slice(&q[..8]);
        let phi = constraint_vector(&q_arr, &params.inner);
        Ok(phi.as_slice().to_vec())
    }

    /// Golfer constraint Jacobian
    #[pyfunction]
    pub fn py_golfer_constraint_jacobian(
        q: Vec<f64>,
        params: &PyGolferParams,
    ) -> PyResult<Vec<Vec<f64>>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 8",
            ));
        }
        let mut q_arr = [0.0; 8];
        q_arr.copy_from_slice(&q[..8]);
        let j = constraint_jacobian(&q_arr, &params.inner);
        let result: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..8).map(|jj| j[(i, jj)]).collect())
            .collect();
        Ok(result)
    }

    /// Golfer constrained accelerations and Lagrange multipliers.
    #[pyfunction]
    #[pyo3(signature = (q, qdot, tau, params, alpha=5.0, beta=5.0))]
    pub fn py_golfer_constrained_dynamics(
        q: Vec<f64>,
        qdot: Vec<f64>,
        tau: Vec<f64>,
        params: &PyGolferParams,
        alpha: f64,
        beta: f64,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        if !(alpha.is_finite() && alpha >= 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "alpha must be finite and non-negative",
            ));
        }
        if !(beta.is_finite() && beta >= 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "beta must be finite and non-negative",
            ));
        }

        let q_arr = to_array_8(q, "q")?;
        let qdot_arr = to_array_8(qdot, "qdot")?;
        let tau_arr = to_array_8(tau, "tau")?;
        let gains = BaumgarteGains { alpha, beta };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            golfer_constrained_accelerations(&q_arr, &qdot_arr, &tau_arr, &params.inner, &gains)
        }));

        match result {
            Ok((qddot, lambda)) => Ok((qddot.as_slice().to_vec(), lambda.as_slice().to_vec())),
            Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "native golfer constrained dynamics failed",
            )),
        }
    }

    /// Project generalized coordinates to the golfer constraint manifold.
    #[pyfunction]
    #[pyo3(signature = (q, params, max_iters=20, tol=1e-8))]
    pub fn py_golfer_project_to_constraints(
        q: Vec<f64>,
        params: &PyGolferParams,
        max_iters: usize,
        tol: f64,
    ) -> PyResult<Vec<f64>> {
        if !(tol.is_finite() && tol > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tol must be finite and positive",
            ));
        }

        let q_arr = to_array_8(q, "q")?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            golfer_project_to_constraints(&q_arr, &params.inner, max_iters, tol)
        }));

        let q_proj = match result {
            Ok(projected) => projected,
            Err(_) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "native golfer constraint projection failed",
                ))
            }
        };

        let residual = constraint_vector(&q_proj, &params.inner).norm();
        if residual > tol {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "native golfer projection did not converge: residual={residual}"
            )));
        }

        Ok(q_proj.to_vec())
    }

    /// Project generalized velocities onto the golfer velocity constraint surface.
    #[pyfunction]
    pub fn py_golfer_project_velocity(
        q: Vec<f64>,
        qdot: Vec<f64>,
        params: &PyGolferParams,
    ) -> PyResult<Vec<f64>> {
        let q_arr = to_array_8(q, "q")?;
        let qdot_arr = to_array_8(qdot, "qdot")?;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            golfer_project_velocity(&q_arr, &qdot_arr, &params.inner)
        }));

        match result {
            Ok(qdot_proj) => Ok(qdot_proj.to_vec()),
            Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "native golfer velocity projection failed",
            )),
        }
    }

    // -------- Jacobians & Ellipsoids --------

    /// Compute double pendulum Jacobians at wrist and tip.
    /// Returns {"wrist": [[j00,j01],[j10,j11]], "tip": [[j00,j01],[j10,j11]]}.
    #[pyfunction]
    pub fn py_double_jacobians(
        theta1: f64,
        phi: f64,
        l1: f64,
        l2: f64,
    ) -> PyResult<HashMap<String, Vec<Vec<f64>>>> {
        let (j_wrist, j_tip) = crate::jacobians::jacobian_double(theta1, phi, l1, l2);
        let mut result = HashMap::new();
        result.insert(
            "wrist".to_string(),
            vec![
                vec![j_wrist[(0, 0)], j_wrist[(0, 1)]],
                vec![j_wrist[(1, 0)], j_wrist[(1, 1)]],
            ],
        );
        result.insert(
            "tip".to_string(),
            vec![
                vec![j_tip[(0, 0)], j_tip[(0, 1)]],
                vec![j_tip[(1, 0)], j_tip[(1, 1)]],
            ],
        );
        Ok(result)
    }

    /// Compute double pendulum ellipsoid data.
    /// Returns {"wrist": {...}, "tip": {...}} with mobility and force semi-axes.
    #[pyfunction]
    pub fn py_double_ellipsoids(
        theta1: f64,
        phi: f64,
        l1: f64,
        l2: f64,
    ) -> PyResult<HashMap<String, HashMap<String, Vec<f64>>>> {
        let (e_wrist, e_tip) = crate::jacobians::ellipsoids_double(theta1, phi, l1, l2);
        let mut result = HashMap::new();
        result.insert("wrist".to_string(), ellipsoid_to_map(&e_wrist));
        result.insert("tip".to_string(), ellipsoid_to_map(&e_tip));
        Ok(result)
    }

    /// Compute triple pendulum Jacobians at wrist1, wrist2, and tip.
    #[pyfunction]
    pub fn py_triple_jacobians(
        theta1: f64,
        phi1: f64,
        phi2: f64,
        l1: f64,
        l2: f64,
        l3: f64,
    ) -> PyResult<HashMap<String, Vec<Vec<f64>>>> {
        let (j_w1, j_w2, j_tip) =
            crate::jacobians::jacobian_triple(theta1, phi1, phi2, l1, l2, l3);
        let mut result = HashMap::new();
        for (name, j) in [("wrist1", j_w1), ("wrist2", j_w2), ("tip", j_tip)] {
            result.insert(
                name.to_string(),
                vec![
                    vec![j[(0, 0)], j[(0, 1)], j[(0, 2)]],
                    vec![j[(1, 0)], j[(1, 1)], j[(1, 2)]],
                ],
            );
        }
        Ok(result)
    }

    /// Compute triple pendulum ellipsoid data.
    #[pyfunction]
    pub fn py_triple_ellipsoids(
        theta1: f64,
        phi1: f64,
        phi2: f64,
        l1: f64,
        l2: f64,
        l3: f64,
    ) -> PyResult<HashMap<String, HashMap<String, Vec<f64>>>> {
        let (e_w1, e_w2, e_tip) =
            crate::jacobians::ellipsoids_triple(theta1, phi1, phi2, l1, l2, l3);
        let mut result = HashMap::new();
        result.insert("wrist1".to_string(), ellipsoid_to_map(&e_w1));
        result.insert("wrist2".to_string(), ellipsoid_to_map(&e_w2));
        result.insert("tip".to_string(), ellipsoid_to_map(&e_tip));
        Ok(result)
    }

    fn ellipsoid_to_map(
        e: &crate::jacobians::EllipsoidResult,
    ) -> HashMap<String, Vec<f64>> {
        let mut m = HashMap::new();
        m.insert(
            "directions".to_string(),
            vec![
                e.directions[(0, 0)],
                e.directions[(1, 0)],
                e.directions[(0, 1)],
                e.directions[(1, 1)],
            ],
        );
        m.insert(
            "mob_semi_axes".to_string(),
            vec![e.mob_semi_axes[0], e.mob_semi_axes[1]],
        );
        if let Some(ref fsa) = e.force_semi_axes {
            m.insert("force_semi_axes".to_string(), vec![fsa[0], fsa[1]]);
        }
        m.insert(
            "singular_values".to_string(),
            vec![e.singular_values[0], e.singular_values[1]],
        );
        m
    }

    // -------- CMA-ES optimisation --------

    /// Run CMA-ES optimization on torque polynomial coefficients (Python interface).
    ///
    /// This is a basic Python wrapper that runs a CMA-ES with a simple synthetic
    /// objective function for testing. For real usage, use the Rust API.
    #[pyfunction]
    #[pyo3(signature = (n_joints, n_coeffs_per_joint, initial_coeffs, max_iterations=500))]
    pub fn py_cmaes_optimize_torque(
        n_joints: usize,
        n_coeffs_per_joint: usize,
        initial_coeffs: Vec<f64>,
        max_iterations: usize,
    ) -> PyResult<PyCmaEsResult> {
        let config = crate::cmaes::CmaEsConfig {
            population_size: 0, // auto
            max_iterations,
            initial_sigma: 0.3,
            target_fitness: None,
            fitness_tolerance: 1e-12,
        };

        let result = crate::cmaes::optimize_torque_coefficients(
            n_joints,
            n_coeffs_per_joint,
            &initial_coeffs,
            |coeffs| coeffs.iter().map(|c| c * c).sum::<f64>(),
            &config,
        );

        Ok(PyCmaEsResult {
            best_solution: result.best_solution,
            best_fitness: result.best_fitness,
            fitness_history: result.fitness_history,
            generations: result.generations,
            evaluations: result.evaluations,
        })
    }
}

// ---------------------------------------------------------------------------
// WASM bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
pub mod wasm {
    use crate::bindings::state::wasm::{WasmCmaEsConfig, WasmCmaEsResult};
    use wasm_bindgen::prelude::*;

    /// Run CMA-ES optimization on torque coefficients (WASM interface).
    #[wasm_bindgen]
    pub fn wasm_cmaes_optimize_torque(
        n_joints: usize,
        n_coeffs_per_joint: usize,
        initial_coeffs: &[f64],
        config: &WasmCmaEsConfig,
    ) -> Result<WasmCmaEsResult, JsValue> {
        if initial_coeffs.len() != n_joints * n_coeffs_per_joint {
            return Err(JsValue::from_str("initial_coeffs length mismatch"));
        }

        let rust_config = crate::cmaes::CmaEsConfig {
            population_size: config.population_size,
            max_iterations: config.max_iterations,
            initial_sigma: config.initial_sigma,
            target_fitness: config.target_fitness,
            fitness_tolerance: config.fitness_tolerance,
        };

        let result = crate::cmaes::optimize_torque_coefficients(
            n_joints,
            n_coeffs_per_joint,
            initial_coeffs,
            |coeffs| coeffs.iter().map(|c| c * c).sum::<f64>(),
            &rust_config,
        );

        Ok(WasmCmaEsResult {
            best_solution: result.best_solution,
            best_fitness: result.best_fitness,
            fitness_history: result.fitness_history,
            generations: result.generations,
            evaluations: result.evaluations,
        })
    }
}
