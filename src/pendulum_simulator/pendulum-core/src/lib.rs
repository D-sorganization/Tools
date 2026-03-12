//! Pendulum Core: A shared physics kernel for pendulum golf swing simulators.
//!
//! This library provides high-performance physics implementations for:
//! - Double pendulum (2-DOF)
//! - Triple pendulum (3-DOF)
//! - Golfer upper body (8-DOF with 4 constraints)
//!
//! Compiled as a native library (PyO3 for Python FFI) and WASM (for web apps).
//!
//! # Features
//!
//! - `python`: Compile with PyO3 bindings for Python FFI
//! - `wasm`: Compile with wasm-bindgen for WASM targets
//! - `serde`: Enable serialization support via serde

pub mod batch;
pub mod cmaes;
pub mod double;
pub mod golfer;
pub mod golfer_constraints;
pub mod integrator;
pub mod triple;
pub mod types;

pub use double::{
    coriolis as double_coriolis, equations_of_motion as double_equations_of_motion,
    forward_kinematics as double_forward_kinematics, friction_torque as double_friction_torque,
    gravity_vector as double_gravity_vector, jacobian_club_tip, jacobian_wrist,
    mass_matrix as double_mass_matrix,
};

pub use triple::{
    coriolis as triple_coriolis, equations_of_motion as triple_equations_of_motion,
    forward_kinematics as triple_forward_kinematics, friction_torque as triple_friction_torque,
    gravity_vector as triple_gravity_vector, jacobian_joint1, jacobian_joint2, jacobian_joint3,
    mass_matrix as triple_mass_matrix,
};

pub use golfer::{
    analytical_fk_jacobians, constraint_jacobian, constraint_vector,
    forward_kinematics as golfer_forward_kinematics, friction_torque as golfer_friction_torque,
    gravity_vector as golfer_gravity_vector, mass_matrix as golfer_mass_matrix,
};

pub use golfer_constraints::{
    constrained_accelerations, constraint_acceleration_bias, project_to_constraints,
    project_velocity, BaumgarteGains,
};

pub use integrator::{
    integrate_double_pendulum, integrate_golfer, integrate_triple_pendulum, RK45Config,
};

pub use cmaes::{optimize, optimize_torque_coefficients, CmaEsConfig, CmaEsResult};

pub use types::{
    DoubleFKResult, DoublePendulumParams, GolferFKResult, GolferParams, TripleFKResult,
    TriplePendulumParams, Vec2,
};

// Re-export commonly used items
pub use nalgebra::{SMatrix, SVector};

#[cfg(feature = "python")]
pub mod py_bindings {
    //! Python FFI bindings via PyO3.

    use crate::golfer_constraints::{
        constrained_accelerations as golfer_constrained_accelerations,
        project_to_constraints as golfer_project_to_constraints,
        project_velocity as golfer_project_velocity, BaumgarteGains,
    };
    use crate::types::*;
    use crate::*;
    use pyo3::prelude::*;
    use std::collections::HashMap;

    fn to_array_8(values: Vec<f64>, name: &str) -> PyResult<[f64; 8]> {
        values.try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(format!("{name} must have length 8"))
        })
    }

    /// Python wrapper for DoublePendulumParams
    #[pyclass]
    #[derive(Clone)]
    pub struct PyDoublePendulumParams {
        pub inner: DoublePendulumParams,
    }

    #[pymethods]
    impl PyDoublePendulumParams {
        #[new]
        #[pyo3(signature = (m1, m2, l1, l2, g=9.81, friction1=0.0, friction2=0.0, m_clubhead=0.0))]
        pub fn new(
            m1: f64,
            m2: f64,
            l1: f64,
            l2: f64,
            g: f64,
            friction1: f64,
            friction2: f64,
            m_clubhead: f64,
        ) -> Self {
            PyDoublePendulumParams {
                inner: DoublePendulumParams {
                    m1,
                    m2,
                    m_clubhead,
                    l1,
                    l2,
                    g,
                    friction1,
                    friction2,
                },
            }
        }

        pub fn validate(&self) -> PyResult<()> {
            self.inner
                .validate()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
        }
    }

    /// Double pendulum mass matrix
    #[pyfunction]
    pub fn py_double_mass_matrix(
        q: Vec<f64>,
        params: &PyDoublePendulumParams,
    ) -> PyResult<Vec<Vec<f64>>> {
        if q.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 2",
            ));
        }
        let q_arr = [q[0], q[1]];
        let m = double_mass_matrix(&q_arr, &params.inner);
        Ok(vec![vec![m[(0, 0)], m[(0, 1)]], vec![m[(1, 0)], m[(1, 1)]]])
    }

    /// Double pendulum gravity vector
    #[pyfunction]
    pub fn py_double_gravity_vector(
        q: Vec<f64>,
        params: &PyDoublePendulumParams,
    ) -> PyResult<Vec<f64>> {
        if q.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 2",
            ));
        }
        let q_arr = [q[0], q[1]];
        let g = double_gravity_vector(&q_arr, &params.inner);
        Ok(g.as_slice().to_vec())
    }

    /// Double pendulum Coriolis vector
    #[pyfunction]
    pub fn py_double_coriolis(
        q: Vec<f64>,
        qdot: Vec<f64>,
        params: &PyDoublePendulumParams,
    ) -> PyResult<Vec<f64>> {
        if q.len() != 2 || qdot.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q and qdot must have length 2",
            ));
        }
        let q_arr = [q[0], q[1]];
        let qdot_arr = [qdot[0], qdot[1]];
        let c = double_coriolis(&q_arr, &qdot_arr, &params.inner);
        Ok(c.as_slice().to_vec())
    }

    /// Double pendulum friction torque
    #[pyfunction]
    pub fn py_double_friction_torque(
        qdot: Vec<f64>,
        params: &PyDoublePendulumParams,
    ) -> PyResult<Vec<f64>> {
        if qdot.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "qdot must have length 2",
            ));
        }
        let qdot_arr = [qdot[0], qdot[1]];
        let f = double_friction_torque(&qdot_arr, &params.inner);
        Ok(f.as_slice().to_vec())
    }

    /// Double pendulum equations of motion (with friction)
    #[pyfunction]
    pub fn py_double_equations_of_motion(
        q: Vec<f64>,
        qdot: Vec<f64>,
        tau: Vec<f64>,
        params: &PyDoublePendulumParams,
    ) -> PyResult<Vec<f64>> {
        if q.len() != 2 || qdot.len() != 2 || tau.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q, qdot, and tau must have length 2",
            ));
        }
        let qddot = double_equations_of_motion(
            &[q[0], q[1]],
            &[qdot[0], qdot[1]],
            &[tau[0], tau[1]],
            &params.inner,
        );
        Ok(qddot.to_vec())
    }

    /// Double pendulum forward kinematics
    #[pyfunction]
    pub fn py_double_forward_kinematics(
        q: Vec<f64>,
        params: &PyDoublePendulumParams,
    ) -> PyResult<HashMap<String, f64>> {
        if q.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 2",
            ));
        }
        let q_arr = [q[0], q[1]];
        let fk = double_forward_kinematics(&q_arr, &params.inner);
        let mut result = HashMap::new();
        result.insert("wrist_x".to_string(), fk.wrist.0);
        result.insert("wrist_y".to_string(), fk.wrist.1);
        result.insert("club_tip_x".to_string(), fk.club_tip.0);
        result.insert("club_tip_y".to_string(), fk.club_tip.1);
        result.insert("theta1".to_string(), fk.theta1);
        result.insert("theta2".to_string(), fk.theta2);
        Ok(result)
    }

    /// Python wrapper for TriplePendulumParams
    #[pyclass]
    #[derive(Clone)]
    pub struct PyTriplePendulumParams {
        pub inner: TriplePendulumParams,
    }

    #[pymethods]
    impl PyTriplePendulumParams {
        #[new]
        #[pyo3(signature = (m1, m2, m3, l1, l2, l3, g=9.81, friction1=0.0, friction2=0.0, friction3=0.0))]
        pub fn new(
            m1: f64,
            m2: f64,
            m3: f64,
            l1: f64,
            l2: f64,
            l3: f64,
            g: f64,
            friction1: f64,
            friction2: f64,
            friction3: f64,
        ) -> Self {
            PyTriplePendulumParams {
                inner: TriplePendulumParams {
                    masses: [m1, m2, m3],
                    lengths: [l1, l2, l3],
                    g,
                    friction: [friction1, friction2, friction3],
                },
            }
        }

        pub fn validate(&self) -> PyResult<()> {
            self.inner
                .validate()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
        }
    }

    /// Triple pendulum mass matrix
    #[pyfunction]
    pub fn py_triple_mass_matrix(
        q: Vec<f64>,
        params: &PyTriplePendulumParams,
    ) -> PyResult<Vec<Vec<f64>>> {
        if q.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 3",
            ));
        }
        let q_arr = [q[0], q[1], q[2]];
        let m = triple_mass_matrix(&q_arr, &params.inner);
        Ok(vec![
            vec![m[(0, 0)], m[(0, 1)], m[(0, 2)]],
            vec![m[(1, 0)], m[(1, 1)], m[(1, 2)]],
            vec![m[(2, 0)], m[(2, 1)], m[(2, 2)]],
        ])
    }

    /// Triple pendulum gravity vector
    #[pyfunction]
    pub fn py_triple_gravity_vector(
        q: Vec<f64>,
        params: &PyTriplePendulumParams,
    ) -> PyResult<Vec<f64>> {
        if q.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 3",
            ));
        }
        let q_arr = [q[0], q[1], q[2]];
        let g = triple_gravity_vector(&q_arr, &params.inner);
        Ok(g.as_slice().to_vec())
    }

    /// Triple pendulum Coriolis vector
    #[pyfunction]
    pub fn py_triple_coriolis(
        q: Vec<f64>,
        qdot: Vec<f64>,
        params: &PyTriplePendulumParams,
    ) -> PyResult<Vec<f64>> {
        if q.len() != 3 || qdot.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q and qdot must have length 3",
            ));
        }
        let q_arr = [q[0], q[1], q[2]];
        let qdot_arr = [qdot[0], qdot[1], qdot[2]];
        let c = triple_coriolis(&q_arr, &qdot_arr, &params.inner);
        Ok(c.as_slice().to_vec())
    }

    /// Triple pendulum friction torque
    #[pyfunction]
    pub fn py_triple_friction_torque(
        qdot: Vec<f64>,
        params: &PyTriplePendulumParams,
    ) -> PyResult<Vec<f64>> {
        if qdot.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "qdot must have length 3",
            ));
        }
        let qdot_arr = [qdot[0], qdot[1], qdot[2]];
        let f = triple_friction_torque(&qdot_arr, &params.inner);
        Ok(f.as_slice().to_vec())
    }

    /// Triple pendulum equations of motion (with friction)
    #[pyfunction]
    pub fn py_triple_equations_of_motion(
        q: Vec<f64>,
        qdot: Vec<f64>,
        tau: Vec<f64>,
        params: &PyTriplePendulumParams,
    ) -> PyResult<Vec<f64>> {
        if q.len() != 3 || qdot.len() != 3 || tau.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q, qdot, and tau must have length 3",
            ));
        }
        let qddot = triple_equations_of_motion(
            &[q[0], q[1], q[2]],
            &[qdot[0], qdot[1], qdot[2]],
            &[tau[0], tau[1], tau[2]],
            &params.inner,
        );
        Ok(qddot.to_vec())
    }

    /// Triple pendulum forward kinematics
    #[pyfunction]
    pub fn py_triple_forward_kinematics(
        q: Vec<f64>,
        params: &PyTriplePendulumParams,
    ) -> PyResult<HashMap<String, f64>> {
        if q.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 3",
            ));
        }
        let q_arr = [q[0], q[1], q[2]];
        let fk = triple_forward_kinematics(&q_arr, &params.inner);
        let mut result = HashMap::new();
        result.insert("joint1_x".to_string(), fk.joint1.0);
        result.insert("joint1_y".to_string(), fk.joint1.1);
        result.insert("joint2_x".to_string(), fk.joint2.0);
        result.insert("joint2_y".to_string(), fk.joint2.1);
        result.insert("joint3_x".to_string(), fk.joint3.0);
        result.insert("joint3_y".to_string(), fk.joint3.1);
        result.insert("theta1".to_string(), fk.angles[0]);
        result.insert("theta2".to_string(), fk.angles[1]);
        result.insert("theta3".to_string(), fk.angles[2]);
        Ok(result)
    }

    /// Python wrapper for GolferParams
    #[pyclass]
    #[derive(Clone)]
    pub struct PyGolferParams {
        pub inner: GolferParams,
    }

    #[pymethods]
    impl PyGolferParams {
        #[new]
        #[pyo3(signature = (l_hub, m_hub, d_rs, d_ls, l_r_upper, m_r_upper, l_r_fore, m_r_fore, l_l_upper, m_l_upper, l_l_fore, m_l_fore, l_club, m_club, m_clubhead, grip_right, grip_left, g, friction=None))]
        pub fn new(
            l_hub: f64,
            m_hub: f64,
            d_rs: f64,
            d_ls: f64,
            l_r_upper: f64,
            m_r_upper: f64,
            l_r_fore: f64,
            m_r_fore: f64,
            l_l_upper: f64,
            m_l_upper: f64,
            l_l_fore: f64,
            m_l_fore: f64,
            l_club: f64,
            m_club: f64,
            m_clubhead: f64,
            grip_right: f64,
            grip_left: f64,
            g: f64,
            friction: Option<Vec<f64>>,
        ) -> Self {
            let fric = match friction {
                Some(f) if f.len() >= 7 => [f[0], f[1], f[2], f[3], f[4], f[5], f[6]],
                _ => [0.0; 7],
            };
            PyGolferParams {
                inner: GolferParams {
                    l_hub,
                    m_hub,
                    d_rs,
                    d_ls,
                    l_r_upper,
                    m_r_upper,
                    l_r_fore,
                    m_r_fore,
                    l_l_upper,
                    m_l_upper,
                    l_l_fore,
                    m_l_fore,
                    l_club,
                    m_club,
                    m_clubhead,
                    grip_right,
                    grip_left,
                    g,
                    friction: fric,
                },
            }
        }

        pub fn validate(&self) -> PyResult<()> {
            self.inner
                .validate()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
        }
    }

    /// Golfer mass matrix
    #[pyfunction]
    pub fn py_golfer_mass_matrix(q: Vec<f64>, params: &PyGolferParams) -> PyResult<Vec<Vec<f64>>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 8",
            ));
        }
        let mut q_arr = [0.0; 8];
        q_arr.copy_from_slice(&q[..8]);
        let m = golfer_mass_matrix(&q_arr, &params.inner);
        let result: Vec<Vec<f64>> = (0..8)
            .map(|i| (0..8).map(|j| m[(i, j)]).collect())
            .collect();
        Ok(result)
    }

    /// Golfer gravity vector
    #[pyfunction]
    pub fn py_golfer_gravity_vector(q: Vec<f64>, params: &PyGolferParams) -> PyResult<Vec<f64>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 8",
            ));
        }
        let mut q_arr = [0.0; 8];
        q_arr.copy_from_slice(&q[..8]);
        let g = golfer_gravity_vector(&q_arr, &params.inner);
        Ok(g.as_slice().to_vec())
    }

    /// Golfer friction torque (7 actuated joints + 1 constrained)
    #[pyfunction]
    pub fn py_golfer_friction_torque(
        qdot: Vec<f64>,
        params: &PyGolferParams,
    ) -> PyResult<Vec<f64>> {
        if qdot.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "qdot must have length 8",
            ));
        }
        let mut qdot_arr = [0.0; 8];
        qdot_arr.copy_from_slice(&qdot[..8]);
        let f = golfer_friction_torque(&qdot_arr, &params.inner);
        Ok(f.as_slice().to_vec())
    }

    /// Golfer forward kinematics
    #[pyfunction]
    pub fn py_golfer_forward_kinematics(
        q: Vec<f64>,
        params: &PyGolferParams,
    ) -> PyResult<HashMap<String, Vec<f64>>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must have length 8",
            ));
        }
        let mut q_arr = [0.0; 8];
        q_arr.copy_from_slice(&q[..8]);
        let fk = golfer_forward_kinematics(&q_arr, &params.inner);
        let mut result = HashMap::new();
        result.insert("hub".to_string(), vec![fk.hub.0, fk.hub.1]);
        result.insert(
            "r_shoulder".to_string(),
            vec![fk.r_shoulder.0, fk.r_shoulder.1],
        );
        result.insert("r_elbow".to_string(), vec![fk.r_elbow.0, fk.r_elbow.1]);
        result.insert("r_wrist".to_string(), vec![fk.r_wrist.0, fk.r_wrist.1]);
        result.insert(
            "l_shoulder".to_string(),
            vec![fk.l_shoulder.0, fk.l_shoulder.1],
        );
        result.insert("l_elbow".to_string(), vec![fk.l_elbow.0, fk.l_elbow.1]);
        result.insert("l_wrist".to_string(), vec![fk.l_wrist.0, fk.l_wrist.1]);
        result.insert(
            "club_base".to_string(),
            vec![fk.club_base.0, fk.club_base.1],
        );
        result.insert("club_com".to_string(), vec![fk.club_com.0, fk.club_com.1]);
        result.insert("club_tip".to_string(), vec![fk.club_tip.0, fk.club_tip.1]);
        Ok(result)
    }

    /// Golfer constraint vector
    #[pyfunction]
    pub fn py_golfer_constraint_vector(q: Vec<f64>, params: &PyGolferParams) -> PyResult<Vec<f64>> {
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

    /// Python wrapper for CMA-ES result
    #[pyclass]
    #[derive(Clone)]
    pub struct PyCmaEsResult {
        pub best_solution: Vec<f64>,
        pub best_fitness: f64,
        pub fitness_history: Vec<f64>,
        pub generations: usize,
        pub evaluations: usize,
    }

    #[pymethods]
    impl PyCmaEsResult {
        #[getter]
        pub fn best_solution(&self) -> Vec<f64> {
            self.best_solution.clone()
        }

        #[getter]
        pub fn best_fitness(&self) -> f64 {
            self.best_fitness
        }

        #[getter]
        pub fn fitness_history(&self) -> Vec<f64> {
            self.fitness_history.clone()
        }

        #[getter]
        pub fn generations(&self) -> usize {
            self.generations
        }

        #[getter]
        pub fn evaluations(&self) -> usize {
            self.evaluations
        }
    }

    /// Python wrapper for CMA-ES configuration
    #[pyclass]
    #[derive(Clone)]
    pub struct PyCmaEsConfig {
        pub population_size: usize,
        pub max_iterations: usize,
        pub initial_sigma: f64,
        pub target_fitness: Option<f64>,
        pub fitness_tolerance: f64,
    }

    #[pymethods]
    impl PyCmaEsConfig {
        #[new]
        #[pyo3(signature = (population_size=0, max_iterations=500, initial_sigma=0.3, target_fitness=None, fitness_tolerance=1e-12))]
        pub fn new(
            population_size: usize,
            max_iterations: usize,
            initial_sigma: f64,
            target_fitness: Option<f64>,
            fitness_tolerance: f64,
        ) -> Self {
            PyCmaEsConfig {
                population_size,
                max_iterations,
                initial_sigma,
                target_fitness,
                fitness_tolerance,
            }
        }
    }

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

    /// Module initialization
    #[pymodule]
    pub fn pendulum_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyDoublePendulumParams>()?;
        m.add_class::<PyTriplePendulumParams>()?;
        m.add_class::<PyGolferParams>()?;
        m.add_class::<PyCmaEsConfig>()?;
        m.add_class::<PyCmaEsResult>()?;
        m.add_function(wrap_pyfunction!(py_double_mass_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(py_double_gravity_vector, m)?)?;
        m.add_function(wrap_pyfunction!(py_double_coriolis, m)?)?;
        m.add_function(wrap_pyfunction!(py_double_friction_torque, m)?)?;
        m.add_function(wrap_pyfunction!(py_double_equations_of_motion, m)?)?;
        m.add_function(wrap_pyfunction!(py_double_forward_kinematics, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_mass_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_gravity_vector, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_coriolis, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_friction_torque, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_equations_of_motion, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_forward_kinematics, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_mass_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_gravity_vector, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_friction_torque, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_forward_kinematics, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_constraint_vector, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_constraint_jacobian, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_constrained_dynamics, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_project_to_constraints, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_project_velocity, m)?)?;
        m.add_function(wrap_pyfunction!(py_batch_evaluate_double, m)?)?;
        m.add_function(wrap_pyfunction!(py_cmaes_optimize_torque, m)?)?;
        Ok(())
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

#[cfg(feature = "wasm")]
pub mod wasm_bindings {
    //! WASM bindings via wasm-bindgen.

    use crate::types::*;
    use crate::*;
    use wasm_bindgen::prelude::*;

    /// WASM-safe wrapper for DoublePendulumParams
    #[wasm_bindgen]
    pub struct WasmDoublePendulumParams {
        inner: DoublePendulumParams,
    }

    #[wasm_bindgen]
    impl WasmDoublePendulumParams {
        #[wasm_bindgen(constructor)]
        pub fn new(
            m1: f64,
            m2: f64,
            l1: f64,
            l2: f64,
            g: f64,
            friction1: f64,
            friction2: f64,
        ) -> WasmDoublePendulumParams {
            WasmDoublePendulumParams {
                inner: DoublePendulumParams {
                    m1,
                    m2,
                    m_clubhead: 0.0,
                    l1,
                    l2,
                    g,
                    friction1,
                    friction2,
                },
            }
        }

        #[wasm_bindgen(js_name = withClubhead)]
        pub fn with_clubhead(
            m1: f64,
            m2: f64,
            l1: f64,
            l2: f64,
            g: f64,
            friction1: f64,
            friction2: f64,
            m_clubhead: f64,
        ) -> WasmDoublePendulumParams {
            WasmDoublePendulumParams {
                inner: DoublePendulumParams {
                    m1,
                    m2,
                    m_clubhead,
                    l1,
                    l2,
                    g,
                    friction1,
                    friction2,
                },
            }
        }

        pub fn validate(&self) -> Result<(), JsValue> {
            self.inner.validate().map_err(|e| JsValue::from_str(&e))
        }
    }

    /// WASM-safe wrapper for GolferParams
    #[wasm_bindgen]
    pub struct WasmGolferParams {
        inner: GolferParams,
    }

    #[wasm_bindgen]
    impl WasmGolferParams {
        #[wasm_bindgen(constructor)]
        pub fn new(
            l_hub: f64,
            m_hub: f64,
            d_rs: f64,
            d_ls: f64,
            l_r_upper: f64,
            m_r_upper: f64,
            l_r_fore: f64,
            m_r_fore: f64,
            l_l_upper: f64,
            m_l_upper: f64,
            l_l_fore: f64,
            m_l_fore: f64,
            l_club: f64,
            m_club: f64,
            m_clubhead: f64,
            grip_right: f64,
            grip_left: f64,
            g: f64,
        ) -> WasmGolferParams {
            WasmGolferParams {
                inner: GolferParams {
                    l_hub,
                    m_hub,
                    d_rs,
                    d_ls,
                    l_r_upper,
                    m_r_upper,
                    l_r_fore,
                    m_r_fore,
                    l_l_upper,
                    m_l_upper,
                    l_l_fore,
                    m_l_fore,
                    l_club,
                    m_club,
                    m_clubhead,
                    grip_right,
                    grip_left,
                    g,
                    friction: [0.0; 7],
                },
            }
        }

        pub fn validate(&self) -> Result<(), JsValue> {
            self.inner.validate().map_err(|e| JsValue::from_str(&e))
        }
    }

    /// Double pendulum mass matrix (WASM)
    #[wasm_bindgen]
    pub fn wasm_double_mass_matrix(
        q: &[f64],
        params: &WasmDoublePendulumParams,
    ) -> Result<Vec<f64>, JsValue> {
        if q.len() != 2 {
            return Err(JsValue::from_str("q must have length 2"));
        }
        let q_arr = [q[0], q[1]];
        let m = double_mass_matrix(&q_arr, &params.inner);
        Ok(vec![m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]])
    }

    /// Double pendulum gravity vector (WASM)
    #[wasm_bindgen]
    pub fn wasm_double_gravity_vector(
        q: &[f64],
        params: &WasmDoublePendulumParams,
    ) -> Result<Vec<f64>, JsValue> {
        if q.len() != 2 {
            return Err(JsValue::from_str("q must have length 2"));
        }
        let q_arr = [q[0], q[1]];
        let g = double_gravity_vector(&q_arr, &params.inner);
        Ok(g.as_slice().to_vec())
    }

    /// Golfer mass matrix (WASM)
    #[wasm_bindgen]
    pub fn wasm_golfer_mass_matrix(
        q: &[f64],
        params: &WasmGolferParams,
    ) -> Result<Vec<f64>, JsValue> {
        if q.len() != 8 {
            return Err(JsValue::from_str("q must have length 8"));
        }
        let mut q_arr = [0.0; 8];
        q_arr.copy_from_slice(&q[..8]);
        let m = golfer_mass_matrix(&q_arr, &params.inner);
        let result: Vec<f64> = (0..8)
            .flat_map(|i| (0..8).map(move |j| m[(i, j)]))
            .collect();
        Ok(result)
    }

    /// Golfer forward kinematics (WASM)
    #[wasm_bindgen]
    pub fn wasm_golfer_forward_kinematics(
        q: &[f64],
        params: &WasmGolferParams,
    ) -> Result<Vec<f64>, JsValue> {
        if q.len() != 8 {
            return Err(JsValue::from_str("q must have length 8"));
        }
        let mut q_arr = [0.0; 8];
        q_arr.copy_from_slice(&q[..8]);
        let fk = golfer_forward_kinematics(&q_arr, &params.inner);
        Ok(vec![
            fk.hub.0,
            fk.hub.1,
            fk.r_shoulder.0,
            fk.r_shoulder.1,
            fk.r_elbow.0,
            fk.r_elbow.1,
            fk.r_wrist.0,
            fk.r_wrist.1,
            fk.l_shoulder.0,
            fk.l_shoulder.1,
            fk.l_elbow.0,
            fk.l_elbow.1,
            fk.l_wrist.0,
            fk.l_wrist.1,
            fk.club_base.0,
            fk.club_base.1,
            fk.club_com.0,
            fk.club_com.1,
            fk.club_tip.0,
            fk.club_tip.1,
        ])
    }

    /// WASM-safe wrapper for CMA-ES result
    #[wasm_bindgen]
    pub struct WasmCmaEsResult {
        best_solution: Vec<f64>,
        best_fitness: f64,
        fitness_history: Vec<f64>,
        generations: usize,
        evaluations: usize,
    }

    #[wasm_bindgen]
    impl WasmCmaEsResult {
        #[wasm_bindgen(getter)]
        pub fn best_solution(&self) -> Vec<f64> {
            self.best_solution.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn best_fitness(&self) -> f64 {
            self.best_fitness
        }

        #[wasm_bindgen(getter)]
        pub fn fitness_history(&self) -> Vec<f64> {
            self.fitness_history.clone()
        }

        #[wasm_bindgen(getter)]
        pub fn generations(&self) -> usize {
            self.generations
        }

        #[wasm_bindgen(getter)]
        pub fn evaluations(&self) -> usize {
            self.evaluations
        }
    }

    /// WASM-safe wrapper for CMA-ES configuration
    #[wasm_bindgen]
    pub struct WasmCmaEsConfig {
        population_size: usize,
        max_iterations: usize,
        initial_sigma: f64,
        target_fitness: Option<f64>,
        fitness_tolerance: f64,
    }

    #[wasm_bindgen]
    impl WasmCmaEsConfig {
        #[wasm_bindgen(constructor)]
        pub fn new(
            population_size: usize,
            max_iterations: usize,
            initial_sigma: f64,
        ) -> WasmCmaEsConfig {
            WasmCmaEsConfig {
                population_size: if population_size == 0 {
                    0
                } else {
                    population_size
                },
                max_iterations,
                initial_sigma,
                target_fitness: None,
                fitness_tolerance: 1e-12,
            }
        }

        #[wasm_bindgen(js_name = withTargetFitness)]
        pub fn with_target_fitness(
            population_size: usize,
            max_iterations: usize,
            initial_sigma: f64,
            target_fitness: f64,
        ) -> WasmCmaEsConfig {
            WasmCmaEsConfig {
                population_size,
                max_iterations,
                initial_sigma,
                target_fitness: Some(target_fitness),
                fitness_tolerance: 1e-12,
            }
        }
    }

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
