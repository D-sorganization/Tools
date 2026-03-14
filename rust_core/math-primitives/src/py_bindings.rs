//! PyO3 bindings for math_primitives.
//!
//! Exposes rotation conversions, quaternion ops, Pose6DOF, Transform6DOF,
//! and geometric primitives to Python.
//!
//! ## Zero-Copy NumPy Integration (#1253)
//!
//! Functions suffixed with `_np` accept and return NumPy arrays directly
//! via `PyReadonlyArray` / `PyArray`, avoiding per-call data copies.
//! Batch variants (e.g. `batch_euler_to_quaternion_np`) process N items
//! in a single call for maximum throughput.

#[cfg(feature = "python")]
pub mod py_math {
    use pyo3::prelude::*;

    use crate::geometry;
    use crate::quaternion;
    use crate::rotation;
    use crate::transform;

    // NumPy imports (available when `python` feature is active)
    use numpy::ndarray::{Array1, Array2};
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

    // -----------------------------------------------------------------------
    // Rotation conversions (original list-based API — backwards compatible)
    // -----------------------------------------------------------------------

    #[pyfunction]
    #[pyo3(name = "euler_to_rotation_matrix")]
    fn py_euler_to_rotation_matrix(euler: [f64; 3]) -> Vec<Vec<f64>> {
        let r = rotation::euler_to_rotation_matrix(&euler);
        (0..3)
            .map(|i| (0..3).map(|j| r[(i, j)]).collect())
            .collect()
    }

    #[pyfunction]
    #[pyo3(name = "rotation_matrix_to_euler")]
    fn py_rotation_matrix_to_euler(r: Vec<Vec<f64>>) -> [f64; 3] {
        let mat = nalgebra::Matrix3::new(
            r[0][0], r[0][1], r[0][2], r[1][0], r[1][1], r[1][2], r[2][0], r[2][1], r[2][2],
        );
        rotation::rotation_matrix_to_euler(&mat)
    }

    #[pyfunction]
    #[pyo3(name = "euler_to_quaternion")]
    fn py_euler_to_quaternion(euler: [f64; 3]) -> [f64; 4] {
        let q = rotation::euler_to_quaternion(&euler);
        [q.w, q.x, q.y, q.z]
    }

    #[pyfunction]
    #[pyo3(name = "quaternion_to_euler")]
    fn py_quaternion_to_euler(q: [f64; 4]) -> [f64; 3] {
        let qv = quaternion::Quaternion::new(q[0], q[1], q[2], q[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        rotation::quaternion_to_euler(&qv)
    }

    #[pyfunction]
    #[pyo3(name = "quaternion_to_rotation_matrix")]
    fn py_quaternion_to_rotation_matrix(q: [f64; 4]) -> Vec<Vec<f64>> {
        let qv = quaternion::Quaternion::new(q[0], q[1], q[2], q[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        let r = rotation::quaternion_to_rotation_matrix(&qv);
        (0..3)
            .map(|i| (0..3).map(|j| r[(i, j)]).collect())
            .collect()
    }

    #[pyfunction]
    #[pyo3(name = "axis_angle_to_rotation_matrix")]
    fn py_axis_angle_to_rotation_matrix(axis: [f64; 3], angle: f64) -> Vec<Vec<f64>> {
        let r = rotation::axis_angle_to_rotation_matrix(&axis, angle);
        (0..3)
            .map(|i| (0..3).map(|j| r[(i, j)]).collect())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Rotation conversions — NumPy zero-copy variants (#1253)
    // -----------------------------------------------------------------------

    /// euler_to_rotation_matrix_np(euler: np.ndarray[3]) -> np.ndarray[3, 3]
    #[pyfunction]
    #[pyo3(name = "euler_to_rotation_matrix_np")]
    fn py_euler_to_rotation_matrix_np<'py>(
        py: Python<'py>,
        euler: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let e = euler.as_array();
        let r = rotation::euler_to_rotation_matrix(&[e[0], e[1], e[2]]);
        let mut out = Array2::<f64>::zeros((3, 3));
        for i in 0..3 {
            for j in 0..3 {
                out[[i, j]] = r[(i, j)];
            }
        }
        out.into_pyarray(py)
    }

    /// rotation_matrix_to_euler_np(r: np.ndarray[3, 3]) -> np.ndarray[3]
    #[pyfunction]
    #[pyo3(name = "rotation_matrix_to_euler_np")]
    fn py_rotation_matrix_to_euler_np<'py>(
        py: Python<'py>,
        r: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let arr = r.as_array();
        let mat = nalgebra::Matrix3::new(
            arr[[0, 0]],
            arr[[0, 1]],
            arr[[0, 2]],
            arr[[1, 0]],
            arr[[1, 1]],
            arr[[1, 2]],
            arr[[2, 0]],
            arr[[2, 1]],
            arr[[2, 2]],
        );
        let euler = rotation::rotation_matrix_to_euler(&mat);
        Array1::from_vec(euler.to_vec()).into_pyarray(py)
    }

    /// euler_to_quaternion_np(euler: np.ndarray[3]) -> np.ndarray[4]
    #[pyfunction]
    #[pyo3(name = "euler_to_quaternion_np")]
    fn py_euler_to_quaternion_np<'py>(
        py: Python<'py>,
        euler: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let e = euler.as_array();
        let q = rotation::euler_to_quaternion(&[e[0], e[1], e[2]]);
        Array1::from_vec(vec![q.w, q.x, q.y, q.z]).into_pyarray(py)
    }

    /// quaternion_to_euler_np(q: np.ndarray[4]) -> np.ndarray[3]
    #[pyfunction]
    #[pyo3(name = "quaternion_to_euler_np")]
    fn py_quaternion_to_euler_np<'py>(
        py: Python<'py>,
        q: PyReadonlyArray1<'py, f64>,
    ) -> Bound<'py, PyArray1<f64>> {
        let a = q.as_array();
        let qv = quaternion::Quaternion::new(a[0], a[1], a[2], a[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        let euler = rotation::quaternion_to_euler(&qv);
        Array1::from_vec(euler.to_vec()).into_pyarray(py)
    }

    // -----------------------------------------------------------------------
    // Batch NumPy operations (#1253)
    // -----------------------------------------------------------------------

    /// batch_euler_to_quaternion_np(eulers: np.ndarray[N,3]) -> np.ndarray[N,4]
    ///
    /// Convert N euler angle triples to quaternions in a single call.
    /// Zero-copy input, returns a freshly allocated output array.
    #[pyfunction]
    #[pyo3(name = "batch_euler_to_quaternion_np")]
    fn py_batch_euler_to_quaternion_np<'py>(
        py: Python<'py>,
        eulers: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let arr = eulers.as_array();
        let n = arr.nrows();
        let mut out = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let q = rotation::euler_to_quaternion(&[arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]]);
            out[[i, 0]] = q.w;
            out[[i, 1]] = q.x;
            out[[i, 2]] = q.y;
            out[[i, 3]] = q.z;
        }
        out.into_pyarray(py)
    }

    /// batch_quaternion_to_euler_np(quats: np.ndarray[N,4]) -> np.ndarray[N,3]
    ///
    /// Convert N quaternions to euler angles in a single call.
    #[pyfunction]
    #[pyo3(name = "batch_quaternion_to_euler_np")]
    fn py_batch_quaternion_to_euler_np<'py>(
        py: Python<'py>,
        quats: PyReadonlyArray2<'py, f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let arr = quats.as_array();
        let n = arr.nrows();
        let mut out = Array2::<f64>::zeros((n, 3));
        for i in 0..n {
            let qv =
                quaternion::Quaternion::new(arr[[i, 0]], arr[[i, 1]], arr[[i, 2]], arr[[i, 3]])
                    .unwrap_or_else(|_| quaternion::Quaternion::identity());
            let euler = rotation::quaternion_to_euler(&qv);
            out[[i, 0]] = euler[0];
            out[[i, 1]] = euler[1];
            out[[i, 2]] = euler[2];
        }
        out.into_pyarray(py)
    }

    /// batch_quaternion_multiply_np(q1s: np.ndarray[N,4], q2s: np.ndarray[N,4]) -> np.ndarray[N,4]
    ///
    /// Element-wise quaternion multiplication for N pairs.
    #[pyfunction]
    #[pyo3(name = "batch_quaternion_multiply_np")]
    fn py_batch_quaternion_multiply_np<'py>(
        py: Python<'py>,
        q1s: PyReadonlyArray2<'py, f64>,
        q2s: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let a1 = q1s.as_array();
        let a2 = q2s.as_array();
        if a1.nrows() != a2.nrows() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q1s and q2s must have the same number of rows",
            ));
        }
        let n = a1.nrows();
        let mut out = Array2::<f64>::zeros((n, 4));
        for i in 0..n {
            let qa = quaternion::Quaternion::new(a1[[i, 0]], a1[[i, 1]], a1[[i, 2]], a1[[i, 3]])
                .unwrap_or_else(|_| quaternion::Quaternion::identity());
            let qb = quaternion::Quaternion::new(a2[[i, 0]], a2[[i, 1]], a2[[i, 2]], a2[[i, 3]])
                .unwrap_or_else(|_| quaternion::Quaternion::identity());
            let r = qa.multiply(&qb);
            out[[i, 0]] = r.w;
            out[[i, 1]] = r.x;
            out[[i, 2]] = r.y;
            out[[i, 3]] = r.z;
        }
        Ok(out.into_pyarray(py))
    }

    // -----------------------------------------------------------------------
    // Quaternion operations (original API — backwards compatible)
    // -----------------------------------------------------------------------

    #[pyfunction]
    #[pyo3(name = "quaternion_multiply")]
    fn py_quaternion_multiply(q1: [f64; 4], q2: [f64; 4]) -> [f64; 4] {
        let a = quaternion::Quaternion::new(q1[0], q1[1], q1[2], q1[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        let b = quaternion::Quaternion::new(q2[0], q2[1], q2[2], q2[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        let r = a.multiply(&b);
        [r.w, r.x, r.y, r.z]
    }

    #[pyfunction]
    #[pyo3(name = "quaternion_inverse")]
    fn py_quaternion_inverse(q: [f64; 4]) -> [f64; 4] {
        let qv = quaternion::Quaternion::new(q[0], q[1], q[2], q[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        let r = qv.conjugate();
        [r.w, r.x, r.y, r.z]
    }

    #[pyfunction]
    #[pyo3(name = "slerp")]
    fn py_slerp(q1: [f64; 4], q2: [f64; 4], t: f64) -> [f64; 4] {
        let a = quaternion::Quaternion::new(q1[0], q1[1], q1[2], q1[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        let b = quaternion::Quaternion::new(q2[0], q2[1], q2[2], q2[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        let r = a.slerp(&b, t);
        [r.w, r.x, r.y, r.z]
    }

    /// slerp_np(q1: np.ndarray[4], q2: np.ndarray[4], t: float) -> np.ndarray[4]
    #[pyfunction]
    #[pyo3(name = "slerp_np")]
    fn py_slerp_np<'py>(
        py: Python<'py>,
        q1: PyReadonlyArray1<'py, f64>,
        q2: PyReadonlyArray1<'py, f64>,
        t: f64,
    ) -> Bound<'py, PyArray1<f64>> {
        let a1 = q1.as_array();
        let a2 = q2.as_array();
        let qa = quaternion::Quaternion::new(a1[0], a1[1], a1[2], a1[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        let qb = quaternion::Quaternion::new(a2[0], a2[1], a2[2], a2[3])
            .unwrap_or_else(|_| quaternion::Quaternion::identity());
        let r = qa.slerp(&qb, t);
        Array1::from_vec(vec![r.w, r.x, r.y, r.z]).into_pyarray(py)
    }

    // -----------------------------------------------------------------------
    // Pose6DOF
    // -----------------------------------------------------------------------

    #[pyclass(name = "Pose6DOF")]
    #[derive(Clone)]
    struct PyPose6DOF {
        inner: transform::Pose6DOF,
    }

    #[pymethods]
    impl PyPose6DOF {
        #[new]
        fn new(position: [f64; 3], euler_angles: [f64; 3]) -> Self {
            Self {
                inner: transform::Pose6DOF::new(position, euler_angles),
            }
        }

        #[getter]
        fn x(&self) -> f64 {
            self.inner.x()
        }

        #[getter]
        fn y(&self) -> f64 {
            self.inner.y()
        }

        #[getter]
        fn z(&self) -> f64 {
            self.inner.z()
        }

        #[getter]
        fn roll(&self) -> f64 {
            self.inner.roll()
        }

        #[getter]
        fn pitch(&self) -> f64 {
            self.inner.pitch()
        }

        #[getter]
        fn yaw(&self) -> f64 {
            self.inner.yaw()
        }

        #[getter]
        fn position(&self) -> [f64; 3] {
            let p = &self.inner.position;
            [p[0], p[1], p[2]]
        }

        #[getter]
        fn euler_angles(&self) -> [f64; 3] {
            self.inner.euler_angles
        }

        fn translate(&self, offset: [f64; 3]) -> Self {
            Self {
                inner: self.inner.translate(&offset),
            }
        }

        fn compose(&self, other: &PyPose6DOF) -> Self {
            Self {
                inner: self.inner.compose(&other.inner),
            }
        }

        fn inverse(&self) -> Self {
            Self {
                inner: self.inner.inverse(),
            }
        }

        fn transform_point(&self, point: [f64; 3]) -> [f64; 3] {
            self.inner.transform_point(&point)
        }

        fn to_quaternion(&self) -> [f64; 4] {
            let q = self.inner.to_quaternion();
            [q.w, q.x, q.y, q.z]
        }

        /// transform_point_np(point: np.ndarray[3]) -> np.ndarray[3]
        fn transform_point_np<'py>(
            &self,
            py: Python<'py>,
            point: PyReadonlyArray1<'py, f64>,
        ) -> Bound<'py, PyArray1<f64>> {
            let p = point.as_array();
            let result = self.inner.transform_point(&[p[0], p[1], p[2]]);
            Array1::from_vec(result.to_vec()).into_pyarray(py)
        }

        /// batch_transform_points_np(points: np.ndarray[N,3]) -> np.ndarray[N,3]
        ///
        /// Transform N 3D points in a single call using zero-copy input.
        fn batch_transform_points_np<'py>(
            &self,
            py: Python<'py>,
            points: PyReadonlyArray2<'py, f64>,
        ) -> Bound<'py, PyArray2<f64>> {
            let arr = points.as_array();
            let n = arr.nrows();
            let mut out = Array2::<f64>::zeros((n, 3));
            for i in 0..n {
                let result = self
                    .inner
                    .transform_point(&[arr[[i, 0]], arr[[i, 1]], arr[[i, 2]]]);
                out[[i, 0]] = result[0];
                out[[i, 1]] = result[1];
                out[[i, 2]] = result[2];
            }
            out.into_pyarray(py)
        }

        fn __repr__(&self) -> String {
            format!(
                "Pose6DOF(position=[{:.4}, {:.4}, {:.4}], euler=[{:.4}, {:.4}, {:.4}])",
                self.inner.x(),
                self.inner.y(),
                self.inner.z(),
                self.inner.roll(),
                self.inner.pitch(),
                self.inner.yaw()
            )
        }
    }

    // -----------------------------------------------------------------------
    // Geometric primitives — distance
    // -----------------------------------------------------------------------

    #[pyfunction]
    #[pyo3(name = "sphere_sphere_distance")]
    fn py_sphere_sphere_distance(
        center_a: [f64; 3],
        radius_a: f64,
        center_b: [f64; 3],
        radius_b: f64,
    ) -> (f64, [f64; 3], [f64; 3]) {
        let a = geometry::Sphere::new(center_a, radius_a);
        let b = geometry::Sphere::new(center_b, radius_b);
        let r = geometry::sphere_sphere_distance(&a, &b);
        (
            r.distance,
            [r.point_a[0], r.point_a[1], r.point_a[2]],
            [r.point_b[0], r.point_b[1], r.point_b[2]],
        )
    }

    #[pyfunction]
    #[pyo3(name = "check_collision_spheres")]
    fn py_check_collision_spheres(
        center_a: [f64; 3],
        radius_a: f64,
        center_b: [f64; 3],
        radius_b: f64,
        margin: f64,
    ) -> bool {
        let a = geometry::Sphere::new(center_a, radius_a);
        let b = geometry::Sphere::new(center_b, radius_b);
        geometry::check_collision_spheres(&a, &b, margin)
    }

    // -----------------------------------------------------------------------
    // Module registration
    // -----------------------------------------------------------------------

    pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "math_primitives")?;

        // Rotation (original)
        m.add_function(wrap_pyfunction!(py_euler_to_rotation_matrix, &m)?)?;
        m.add_function(wrap_pyfunction!(py_rotation_matrix_to_euler, &m)?)?;
        m.add_function(wrap_pyfunction!(py_euler_to_quaternion, &m)?)?;
        m.add_function(wrap_pyfunction!(py_quaternion_to_euler, &m)?)?;
        m.add_function(wrap_pyfunction!(py_quaternion_to_rotation_matrix, &m)?)?;
        m.add_function(wrap_pyfunction!(py_axis_angle_to_rotation_matrix, &m)?)?;

        // Rotation (NumPy zero-copy)
        m.add_function(wrap_pyfunction!(py_euler_to_rotation_matrix_np, &m)?)?;
        m.add_function(wrap_pyfunction!(py_rotation_matrix_to_euler_np, &m)?)?;
        m.add_function(wrap_pyfunction!(py_euler_to_quaternion_np, &m)?)?;
        m.add_function(wrap_pyfunction!(py_quaternion_to_euler_np, &m)?)?;

        // Batch NumPy operations
        m.add_function(wrap_pyfunction!(py_batch_euler_to_quaternion_np, &m)?)?;
        m.add_function(wrap_pyfunction!(py_batch_quaternion_to_euler_np, &m)?)?;
        m.add_function(wrap_pyfunction!(py_batch_quaternion_multiply_np, &m)?)?;

        // Quaternion (original)
        m.add_function(wrap_pyfunction!(py_quaternion_multiply, &m)?)?;
        m.add_function(wrap_pyfunction!(py_quaternion_inverse, &m)?)?;
        m.add_function(wrap_pyfunction!(py_slerp, &m)?)?;
        m.add_function(wrap_pyfunction!(py_slerp_np, &m)?)?;

        // Pose6DOF
        m.add_class::<PyPose6DOF>()?;

        // Geometry
        m.add_function(wrap_pyfunction!(py_sphere_sphere_distance, &m)?)?;
        m.add_function(wrap_pyfunction!(py_check_collision_spheres, &m)?)?;

        parent.add_submodule(&m)?;
        Ok(())
    }
}
