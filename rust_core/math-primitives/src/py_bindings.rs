//! PyO3 bindings for math_primitives.
//!
//! Exposes rotation conversions, quaternion ops, Pose6DOF, Transform6DOF,
//! and geometric primitives to Python.

#[cfg(feature = "python")]
pub mod py_bindings {
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    use crate::geometry;
    use crate::quaternion;
    use crate::rotation;
    use crate::transform;

    // -----------------------------------------------------------------------
    // Rotation conversions
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
        let qv = quaternion::Quaternion::new(q[0], q[1], q[2], q[3]).unwrap_or_else(|_| quaternion::Quaternion::identity());
        rotation::quaternion_to_euler(&qv)
    }

    #[pyfunction]
    #[pyo3(name = "quaternion_to_rotation_matrix")]
    fn py_quaternion_to_rotation_matrix(q: [f64; 4]) -> Vec<Vec<f64>> {
        let qv = quaternion::Quaternion::new(q[0], q[1], q[2], q[3]).unwrap_or_else(|_| quaternion::Quaternion::identity());
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
    // Quaternion operations
    // -----------------------------------------------------------------------

    #[pyfunction]
    #[pyo3(name = "quaternion_multiply")]
    fn py_quaternion_multiply(q1: [f64; 4], q2: [f64; 4]) -> [f64; 4] {
        let a = quaternion::Quaternion::new(q1[0], q1[1], q1[2], q1[3]).unwrap_or_else(|_| quaternion::Quaternion::identity());
        let b = quaternion::Quaternion::new(q2[0], q2[1], q2[2], q2[3]).unwrap_or_else(|_| quaternion::Quaternion::identity());
        let r = a.multiply(&b);
        [r.w, r.x, r.y, r.z]
    }

    #[pyfunction]
    #[pyo3(name = "quaternion_inverse")]
    fn py_quaternion_inverse(q: [f64; 4]) -> [f64; 4] {
        let qv = quaternion::Quaternion::new(q[0], q[1], q[2], q[3]).unwrap_or_else(|_| quaternion::Quaternion::identity());
        let r = qv.conjugate();
        [r.w, r.x, r.y, r.z]
    }

    #[pyfunction]
    #[pyo3(name = "slerp")]
    fn py_slerp(q1: [f64; 4], q2: [f64; 4], t: f64) -> [f64; 4] {
        let a = quaternion::Quaternion::new(q1[0], q1[1], q1[2], q1[3]).unwrap_or_else(|_| quaternion::Quaternion::identity());
        let b = quaternion::Quaternion::new(q2[0], q2[1], q2[2], q2[3]).unwrap_or_else(|_| quaternion::Quaternion::identity());
        let r = a.slerp(&b, t);
        [r.w, r.x, r.y, r.z]
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

        // Rotation
        m.add_function(wrap_pyfunction!(py_euler_to_rotation_matrix, &m)?)?;
        m.add_function(wrap_pyfunction!(py_rotation_matrix_to_euler, &m)?)?;
        m.add_function(wrap_pyfunction!(py_euler_to_quaternion, &m)?)?;
        m.add_function(wrap_pyfunction!(py_quaternion_to_euler, &m)?)?;
        m.add_function(wrap_pyfunction!(py_quaternion_to_rotation_matrix, &m)?)?;
        m.add_function(wrap_pyfunction!(py_axis_angle_to_rotation_matrix, &m)?)?;

        // Quaternion
        m.add_function(wrap_pyfunction!(py_quaternion_multiply, &m)?)?;
        m.add_function(wrap_pyfunction!(py_quaternion_inverse, &m)?)?;
        m.add_function(wrap_pyfunction!(py_slerp, &m)?)?;

        // Pose6DOF
        m.add_class::<PyPose6DOF>()?;

        // Geometry
        m.add_function(wrap_pyfunction!(py_sphere_sphere_distance, &m)?)?;
        m.add_function(wrap_pyfunction!(py_check_collision_spheres, &m)?)?;

        parent.add_submodule(&m)?;
        Ok(())
    }
}
