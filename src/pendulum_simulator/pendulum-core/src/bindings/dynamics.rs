//! FFI wrappers for dynamics quantities: mass matrix, gravity vector, Coriolis,
//! friction torque, equations of motion, and forward kinematics.

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
pub mod python {
    use crate::bindings::state::python::{
        PyDoublePendulumParams, PyGolferParams, PyTriplePendulumParams,
    };
    use crate::{
        double_coriolis, double_equations_of_motion, double_forward_kinematics,
        double_friction_torque, double_gravity_vector, double_mass_matrix,
        golfer_forward_kinematics, golfer_friction_torque, golfer_gravity_vector,
        golfer_mass_matrix, triple_coriolis, triple_equations_of_motion,
        triple_forward_kinematics, triple_friction_torque, triple_gravity_vector,
        triple_mass_matrix,
    };
    use pyo3::prelude::*;
    use std::collections::HashMap;

    // -------- Double pendulum --------

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

    // -------- Triple pendulum --------

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

    // -------- Golfer (8-DOF) --------

    #[pyfunction]
    pub fn py_golfer_mass_matrix(
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
        let m = golfer_mass_matrix(&q_arr, &params.inner);
        let result: Vec<Vec<f64>> = (0..8)
            .map(|i| (0..8).map(|j| m[(i, j)]).collect())
            .collect();
        Ok(result)
    }

    #[pyfunction]
    pub fn py_golfer_gravity_vector(
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
        let g = golfer_gravity_vector(&q_arr, &params.inner);
        Ok(g.as_slice().to_vec())
    }

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
}

// ---------------------------------------------------------------------------
// WASM bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
pub mod wasm {
    use crate::bindings::state::wasm::{WasmDoublePendulumParams, WasmGolferParams};
    use crate::{double_gravity_vector, double_mass_matrix, golfer_forward_kinematics,
        golfer_mass_matrix};
    use wasm_bindgen::prelude::*;

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
}
