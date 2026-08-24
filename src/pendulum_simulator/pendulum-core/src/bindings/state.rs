//! Parameter/state wrapper structs for Python (PyO3) and WASM (wasm-bindgen) FFI.
//!
//! This module contains the opaque wrapper classes that expose the core
//! `*Params` structs, together with CMA-ES result/config wrappers.

#![allow(dead_code)]

#[cfg(feature = "python")]
pub(crate) fn to_array_8(values: Vec<f64>, name: &str) -> pyo3::PyResult<[f64; 8]> {
    values
        .try_into()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("{name} must have length 8")))
}

// ---------------------------------------------------------------------------
// Python wrappers
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
pub mod python {
    use crate::types::{DoublePendulumParams, GolferParams, TriplePendulumParams};
    use pyo3::prelude::*;

    /// Python wrapper for DoublePendulumParams
    #[pyclass(from_py_object)]
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

    /// Python wrapper for TriplePendulumParams
    #[pyclass(from_py_object)]
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

    /// Python wrapper for GolferParams
    #[pyclass(from_py_object)]
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

    /// Python wrapper for CMA-ES result
    #[pyclass(from_py_object)]
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
    #[pyclass(from_py_object)]
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
}

// ---------------------------------------------------------------------------
// WASM wrappers
// ---------------------------------------------------------------------------

#[cfg(feature = "wasm")]
pub mod wasm {
    use crate::types::{DoublePendulumParams, GolferParams};
    use wasm_bindgen::prelude::*;

    /// WASM-safe wrapper for DoublePendulumParams
    #[wasm_bindgen]
    pub struct WasmDoublePendulumParams {
        pub(crate) inner: DoublePendulumParams,
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
        pub(crate) inner: GolferParams,
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

    /// WASM-safe wrapper for CMA-ES result
    #[wasm_bindgen]
    pub struct WasmCmaEsResult {
        pub(crate) best_solution: Vec<f64>,
        pub(crate) best_fitness: f64,
        pub(crate) fitness_history: Vec<f64>,
        pub(crate) generations: usize,
        pub(crate) evaluations: usize,
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
        pub(crate) population_size: usize,
        pub(crate) max_iterations: usize,
        pub(crate) initial_sigma: f64,
        pub(crate) target_fitness: Option<f64>,
        pub(crate) fitness_tolerance: f64,
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
}
