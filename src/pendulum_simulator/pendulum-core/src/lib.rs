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

pub mod double;
pub mod golfer;
pub mod golfer_constraints;
pub mod integrator;
pub mod triple;
pub mod types;

pub use double::{
    coriolis as double_coriolis, forward_kinematics as double_forward_kinematics,
    gravity_vector as double_gravity_vector, jacobian_club_tip, jacobian_wrist,
    mass_matrix as double_mass_matrix,
};

pub use triple::{
    coriolis as triple_coriolis, forward_kinematics as triple_forward_kinematics,
    gravity_vector as triple_gravity_vector, jacobian_joint1, jacobian_joint2, jacobian_joint3,
    mass_matrix as triple_mass_matrix,
};

pub use golfer::{
    analytical_fk_jacobians, constraint_jacobian, constraint_vector,
    forward_kinematics as golfer_forward_kinematics, gravity_vector as golfer_gravity_vector,
    mass_matrix as golfer_mass_matrix,
};

pub use golfer_constraints::{
    constrained_accelerations, constraint_acceleration_bias, project_to_constraints,
    project_velocity, BaumgarteGains,
};

pub use integrator::{integrate_double_pendulum, integrate_golfer, integrate_triple_pendulum, RK45Config};

pub use types::{
    DoubleFKResult, DoublePendulumParams, GolferFKResult, GolferParams, TripleFKResult,
    TriplePendulumParams, Vec2,
};

// Re-export commonly used items
pub use nalgebra::{SMatrix, SVector};

#[cfg(feature = "python")]
pub mod py_bindings {
    //! Python FFI bindings via PyO3.

    use crate::types::*;
    use crate::*;
    use crate::golfer_constraints::{
        constrained_accelerations as golfer_constrained_accelerations,
        project_to_constraints as golfer_project_to_constraints,
        project_velocity as golfer_project_velocity,
        BaumgarteGains,
    };
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
            self.inner.validate().map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
        }
    }

    /// Double pendulum mass matrix
    #[pyfunction]
    pub fn py_double_mass_matrix(q: Vec<f64>, params: &PyDoublePendulumParams) -> PyResult<Vec<Vec<f64>>> {
        if q.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 2"));
        }
        let q_arr = [q[0], q[1]];
        let m = double_mass_matrix(&q_arr, &params.inner);
        Ok(vec![
            vec![m[(0, 0)], m[(0, 1)]],
            vec![m[(1, 0)], m[(1, 1)]],
        ])
    }

    /// Double pendulum gravity vector
    #[pyfunction]
    pub fn py_double_gravity_vector(q: Vec<f64>, params: &PyDoublePendulumParams) -> PyResult<Vec<f64>> {
        if q.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 2"));
        }
        let q_arr = [q[0], q[1]];
        let g = double_gravity_vector(&q_arr, &params.inner);
        Ok(g.as_slice().to_vec())
    }

    /// Double pendulum Coriolis vector
    #[pyfunction]
    pub fn py_double_coriolis(q: Vec<f64>, qdot: Vec<f64>, params: &PyDoublePendulumParams) -> PyResult<Vec<f64>> {
        if q.len() != 2 || qdot.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("q and qdot must have length 2"));
        }
        let q_arr = [q[0], q[1]];
        let qdot_arr = [qdot[0], qdot[1]];
        let c = double_coriolis(&q_arr, &qdot_arr, &params.inner);
        Ok(c.as_slice().to_vec())
    }

    /// Double pendulum forward kinematics
    #[pyfunction]
    pub fn py_double_forward_kinematics(q: Vec<f64>, params: &PyDoublePendulumParams) -> PyResult<HashMap<String, f64>> {
        if q.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 2"));
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
            self.inner.validate().map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
        }
    }

    /// Triple pendulum mass matrix
    #[pyfunction]
    pub fn py_triple_mass_matrix(
        q: Vec<f64>,
        params: &PyTriplePendulumParams,
    ) -> PyResult<Vec<Vec<f64>>> {
        if q.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 3"));
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
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 3"));
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

    /// Triple pendulum forward kinematics
    #[pyfunction]
    pub fn py_triple_forward_kinematics(
        q: Vec<f64>,
        params: &PyTriplePendulumParams,
    ) -> PyResult<HashMap<String, f64>> {
        if q.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 3"));
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
        #[pyo3(signature = (l_hub, m_hub, d_rs, d_ls, l_r_upper, m_r_upper, l_r_fore, m_r_fore, l_l_upper, m_l_upper, l_l_fore, m_l_fore, l_club, m_club, m_clubhead, grip_right, grip_left, g))]
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
        ) -> Self {
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
                },
            }
        }

        pub fn validate(&self) -> PyResult<()> {
            self.inner.validate().map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
        }
    }

    /// Golfer mass matrix
    #[pyfunction]
    pub fn py_golfer_mass_matrix(q: Vec<f64>, params: &PyGolferParams) -> PyResult<Vec<Vec<f64>>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 8"));
        }
        let mut q_arr = [0.0; 8];
        for i in 0..8 {
            q_arr[i] = q[i];
        }
        let m = golfer_mass_matrix(&q_arr, &params.inner);
        let mut result = Vec::new();
        for i in 0..8 {
            let mut row = Vec::new();
            for j in 0..8 {
                row.push(m[(i, j)]);
            }
            result.push(row);
        }
        Ok(result)
    }

    /// Golfer gravity vector
    #[pyfunction]
    pub fn py_golfer_gravity_vector(q: Vec<f64>, params: &PyGolferParams) -> PyResult<Vec<f64>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 8"));
        }
        let mut q_arr = [0.0; 8];
        for i in 0..8 {
            q_arr[i] = q[i];
        }
        let g = golfer_gravity_vector(&q_arr, &params.inner);
        Ok(g.as_slice().to_vec())
    }

    /// Golfer forward kinematics
    #[pyfunction]
    pub fn py_golfer_forward_kinematics(q: Vec<f64>, params: &PyGolferParams) -> PyResult<HashMap<String, Vec<f64>>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 8"));
        }
        let mut q_arr = [0.0; 8];
        for i in 0..8 {
            q_arr[i] = q[i];
        }
        let fk = golfer_forward_kinematics(&q_arr, &params.inner);
        let mut result = HashMap::new();
        result.insert("hub".to_string(), vec![fk.hub.0, fk.hub.1]);
        result.insert("r_shoulder".to_string(), vec![fk.r_shoulder.0, fk.r_shoulder.1]);
        result.insert("r_elbow".to_string(), vec![fk.r_elbow.0, fk.r_elbow.1]);
        result.insert("r_wrist".to_string(), vec![fk.r_wrist.0, fk.r_wrist.1]);
        result.insert("l_shoulder".to_string(), vec![fk.l_shoulder.0, fk.l_shoulder.1]);
        result.insert("l_elbow".to_string(), vec![fk.l_elbow.0, fk.l_elbow.1]);
        result.insert("l_wrist".to_string(), vec![fk.l_wrist.0, fk.l_wrist.1]);
        result.insert("club_base".to_string(), vec![fk.club_base.0, fk.club_base.1]);
        result.insert("club_com".to_string(), vec![fk.club_com.0, fk.club_com.1]);
        result.insert("club_tip".to_string(), vec![fk.club_tip.0, fk.club_tip.1]);
        Ok(result)
    }

    /// Golfer constraint vector
    #[pyfunction]
    pub fn py_golfer_constraint_vector(q: Vec<f64>, params: &PyGolferParams) -> PyResult<Vec<f64>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 8"));
        }
        let mut q_arr = [0.0; 8];
        for i in 0..8 {
            q_arr[i] = q[i];
        }
        let phi = constraint_vector(&q_arr, &params.inner);
        Ok(phi.as_slice().to_vec())
    }

    /// Golfer constraint Jacobian
    #[pyfunction]
    pub fn py_golfer_constraint_jacobian(q: Vec<f64>, params: &PyGolferParams) -> PyResult<Vec<Vec<f64>>> {
        if q.len() != 8 {
            return Err(pyo3::exceptions::PyValueError::new_err("q must have length 8"));
        }
        let mut q_arr = [0.0; 8];
        for i in 0..8 {
            q_arr[i] = q[i];
        }
        let j = constraint_jacobian(&q_arr, &params.inner);
        let mut result = Vec::new();
        for i in 0..4 {
            let mut row = Vec::new();
            for jj in 0..8 {
                row.push(j[(i, jj)]);
            }
            result.push(row);
        }
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
            golfer_constrained_accelerations(
                &q_arr,
                &qdot_arr,
                &tau_arr,
                &params.inner,
                &gains,
            )
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

    /// Module initialization
    #[pymodule]
    pub fn pendulum_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyDoublePendulumParams>()?;
        m.add_class::<PyTriplePendulumParams>()?;
        m.add_class::<PyGolferParams>()?;
        m.add_function(wrap_pyfunction!(py_double_mass_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(py_double_gravity_vector, m)?)?;
        m.add_function(wrap_pyfunction!(py_double_coriolis, m)?)?;
        m.add_function(wrap_pyfunction!(py_double_forward_kinematics, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_mass_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_gravity_vector, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_coriolis, m)?)?;
        m.add_function(wrap_pyfunction!(py_triple_forward_kinematics, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_mass_matrix, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_gravity_vector, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_forward_kinematics, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_constraint_vector, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_constraint_jacobian, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_constrained_dynamics, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_project_to_constraints, m)?)?;
        m.add_function(wrap_pyfunction!(py_golfer_project_velocity, m)?)?;
        Ok(())
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
        pub fn new(m1: f64, m2: f64, l1: f64, l2: f64, g: f64, friction1: f64, friction2: f64) -> WasmDoublePendulumParams {
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
                },
            }
        }

        pub fn validate(&self) -> Result<(), JsValue> {
            self.inner.validate().map_err(|e| JsValue::from_str(&e))
        }
    }

    /// Double pendulum mass matrix (WASM)
    #[wasm_bindgen]
    pub fn wasm_double_mass_matrix(q: &[f64], params: &WasmDoublePendulumParams) -> Result<Vec<f64>, JsValue> {
        if q.len() != 2 {
            return Err(JsValue::from_str("q must have length 2"));
        }
        let q_arr = [q[0], q[1]];
        let m = double_mass_matrix(&q_arr, &params.inner);
        Ok(vec![m[(0, 0)], m[(0, 1)], m[(1, 0)], m[(1, 1)]])
    }

    /// Double pendulum gravity vector (WASM)
    #[wasm_bindgen]
    pub fn wasm_double_gravity_vector(q: &[f64], params: &WasmDoublePendulumParams) -> Result<Vec<f64>, JsValue> {
        if q.len() != 2 {
            return Err(JsValue::from_str("q must have length 2"));
        }
        let q_arr = [q[0], q[1]];
        let g = double_gravity_vector(&q_arr, &params.inner);
        Ok(g.as_slice().to_vec())
    }

    /// Golfer mass matrix (WASM)
    #[wasm_bindgen]
    pub fn wasm_golfer_mass_matrix(q: &[f64], params: &WasmGolferParams) -> Result<Vec<f64>, JsValue> {
        if q.len() != 8 {
            return Err(JsValue::from_str("q must have length 8"));
        }
        let mut q_arr = [0.0; 8];
        for i in 0..8 {
            q_arr[i] = q[i];
        }
        let m = golfer_mass_matrix(&q_arr, &params.inner);
        let mut result = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                result.push(m[(i, j)]);
            }
        }
        Ok(result)
    }

    /// Golfer forward kinematics (WASM)
    #[wasm_bindgen]
    pub fn wasm_golfer_forward_kinematics(q: &[f64], params: &WasmGolferParams) -> Result<Vec<f64>, JsValue> {
        if q.len() != 8 {
            return Err(JsValue::from_str("q must have length 8"));
        }
        let mut q_arr = [0.0; 8];
        for i in 0..8 {
            q_arr[i] = q[i];
        }
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
}
