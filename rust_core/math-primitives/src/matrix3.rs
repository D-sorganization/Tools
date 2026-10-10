//! 3×3 rotation/transformation matrix.
//!
//! Row-major storage, compatible with standard math conventions.
//!
//! # Design by Contract
//! - `from_quaternion` produces an orthogonal matrix.
//! - `determinant()` of a rotation matrix is always 1.0.
//! - `debug_assert!` rejects NaN inputs.

use serde::{Deserialize, Serialize};

use crate::quaternion::Quaternion;
use crate::types::Vector3;

/// A 3×3 matrix stored in row-major order.
///
/// Used for rotation matrices, inertia tensors, and Jacobians.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(from_py_object))]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
// NOTE: Matrix3 uses [f64; 9], so WASM support is implemented via custom
// accessors rather than exposing the backing array directly.
pub struct Matrix3 {
    /// Row-major elements: `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`
    data: [f64; 9],
}

impl Matrix3 {
    /// Create a matrix from 9 row-major elements.
    ///
    /// # Layout
    /// ```text
    /// | m00 m01 m02 |
    /// | m10 m11 m12 |
    /// | m20 m21 m22 |
    /// ```
    #[must_use]
    pub fn new(data: [f64; 9]) -> Self {
        debug_assert!(
            data.iter().all(|v| !v.is_nan()),
            "Matrix3::new: no element may be NaN"
        );
        Self { data }
    }

    /// The 3×3 identity matrix.
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// The 3×3 zero matrix.
    #[must_use]
    pub const fn zero() -> Self {
        Self { data: [0.0; 9] }
    }

    /// Access element at (row, col), zero-indexed.
    ///
    /// # Contracts
    /// - Precondition: row < 3, col < 3.
    #[must_use]
    pub fn at(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < 3, "Matrix3::at: row must be < 3, got {row}");
        debug_assert!(col < 3, "Matrix3::at: col must be < 3, got {col}");
        self.data[row * 3 + col]
    }

    /// Determinant of the matrix.
    #[must_use]
    pub fn determinant(&self) -> f64 {
        let d = &self.data;
        d[0] * (d[4] * d[8] - d[5] * d[7]) - d[1] * (d[3] * d[8] - d[5] * d[6])
            + d[2] * (d[3] * d[7] - d[4] * d[6])
    }

    /// Transpose of the matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let d = &self.data;
        Self::new([d[0], d[3], d[6], d[1], d[4], d[7], d[2], d[5], d[8]])
    }

    /// Matrix-vector multiplication: M * v.
    #[must_use]
    pub fn mul_vec(&self, v: &Vector3) -> Vector3 {
        let d = &self.data;
        Vector3::new(
            d[0] * v.x + d[1] * v.y + d[2] * v.z,
            d[3] * v.x + d[4] * v.y + d[5] * v.z,
            d[6] * v.x + d[7] * v.y + d[8] * v.z,
        )
    }

    /// Matrix-matrix multiplication: self * other.
    #[must_use]
    pub fn mul_mat(&self, other: &Self) -> Self {
        let a = &self.data;
        let b = &other.data;
        let mut result = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                result[i * 3 + j] =
                    a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
            }
        }
        Self::new(result)
    }

    /// Construct a rotation matrix from a unit quaternion.
    ///
    /// # DRY
    /// Quaternion → Matrix3 conversion is defined here exactly once.
    #[must_use]
    pub fn from_quaternion(q: &Quaternion) -> Self {
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        Self::new([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ])
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(&self, s: f64) -> Self {
        debug_assert!(!s.is_nan(), "Matrix3::scale: scalar must not be NaN");
        let mut result = self.data;
        for v in &mut result {
            *v *= s;
        }
        Self::new(result)
    }

    /// Matrix addition.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let mut result = [0.0; 9];
        for (r, (a, b)) in result
            .iter_mut()
            .zip(self.data.iter().zip(other.data.iter()))
        {
            *r = a + b;
        }
        Self::new(result)
    }
}

impl std::fmt::Display for Matrix3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "|{:.4} {:.4} {:.4}|\n|{:.4} {:.4} {:.4}|\n|{:.4} {:.4} {:.4}|",
            self.data[0],
            self.data[1],
            self.data[2],
            self.data[3],
            self.data[4],
            self.data[5],
            self.data[6],
            self.data[7],
            self.data[8]
        )
    }
}

// ── Python bindings ──────────────────────────────────────────────────────────

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl Matrix3 {
    #[new]
    fn py_new(data: [f64; 9]) -> Self {
        Self::new(data)
    }

    #[staticmethod]
    #[pyo3(name = "identity")]
    fn py_identity() -> Self {
        Self::identity()
    }

    #[staticmethod]
    #[pyo3(name = "from_quaternion")]
    fn py_from_quaternion(q: &Quaternion) -> Self {
        Self::from_quaternion(q)
    }

    #[pyo3(name = "determinant")]
    fn py_determinant(&self) -> f64 {
        self.determinant()
    }

    #[pyo3(name = "transpose")]
    fn py_transpose(&self) -> Self {
        self.transpose()
    }

    #[pyo3(name = "mul_vec")]
    fn py_mul_vec(&self, v: &Vector3) -> Vector3 {
        self.mul_vec(v)
    }

    fn __repr__(&self) -> String {
        format!("{self}")
    }
}

// ── WASM bindings ────────────────────────────────────────────────────────────
// wasm_bindgen cannot handle [f64; 9] directly, so we provide
// individual element access and convenience constructors.

#[cfg(feature = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl Matrix3 {
    /// Create a Matrix3 from 9 row-major values.
    #[wasm_bindgen(constructor)]
    pub fn wasm_new(
        m00: f64,
        m01: f64,
        m02: f64,
        m10: f64,
        m11: f64,
        m12: f64,
        m20: f64,
        m21: f64,
        m22: f64,
    ) -> Self {
        Self::new([m00, m01, m02, m10, m11, m12, m20, m21, m22])
    }

    /// Create the 3×3 identity matrix.
    #[wasm_bindgen(js_name = "identity")]
    pub fn wasm_identity() -> Self {
        Self::identity()
    }

    /// Get element at (row, col), zero-indexed.
    #[wasm_bindgen(js_name = "at")]
    pub fn wasm_at(&self, row: usize, col: usize) -> f64 {
        self.at(row, col)
    }

    /// Get row as a Vector3.
    #[wasm_bindgen(js_name = "getRow")]
    pub fn wasm_get_row(&self, row: usize) -> Vector3 {
        debug_assert!(row < 3, "getRow: row must be < 3");
        let i = row * 3;
        Vector3::new(self.data[i], self.data[i + 1], self.data[i + 2])
    }

    /// Get column as a Vector3.
    #[wasm_bindgen(js_name = "getCol")]
    pub fn wasm_get_col(&self, col: usize) -> Vector3 {
        debug_assert!(col < 3, "getCol: col must be < 3");
        Vector3::new(self.data[col], self.data[3 + col], self.data[6 + col])
    }

    /// Compute the determinant.
    #[wasm_bindgen(js_name = "determinant")]
    pub fn wasm_determinant(&self) -> f64 {
        self.determinant()
    }

    /// Compute the transpose.
    #[wasm_bindgen(js_name = "transpose")]
    pub fn wasm_transpose(&self) -> Self {
        self.transpose()
    }

    /// Multiply this matrix by a vector.
    #[wasm_bindgen(js_name = "mulVec")]
    pub fn wasm_mul_vec(&self, v: &Vector3) -> Vector3 {
        self.mul_vec(v)
    }

    /// Multiply this matrix by another matrix.
    #[wasm_bindgen(js_name = "mulMat")]
    pub fn wasm_mul_mat(&self, other: &Matrix3) -> Matrix3 {
        self.mul_mat(other)
    }

    /// Scale all elements by a scalar.
    #[wasm_bindgen(js_name = "scale")]
    pub fn wasm_scale(&self, s: f64) -> Self {
        self.scale(s)
    }
}

// ── Tests (TDD) ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_matrix() {
        let m = Matrix3::identity();
        assert!((m.at(0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((m.at(1, 1) - 1.0).abs() < f64::EPSILON);
        assert!((m.at(2, 2) - 1.0).abs() < f64::EPSILON);
        assert!(m.at(0, 1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_identity_determinant() {
        let m = Matrix3::identity();
        assert!((m.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_zero_matrix_determinant() {
        let m = Matrix3::zero();
        assert!(m.determinant().abs() < f64::EPSILON);
    }

    #[test]
    fn test_transpose() {
        let m = Matrix3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let t = m.transpose();
        assert!((t.at(0, 1) - 4.0).abs() < f64::EPSILON);
        assert!((t.at(1, 0) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transpose_twice_is_original() {
        let m = Matrix3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let tt = m.transpose().transpose();
        for i in 0..9 {
            assert!((tt.data[i] - m.data[i]).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_mul_vec_identity() {
        let m = Matrix3::identity();
        let v = Vector3::new(1.0, 2.0, 3.0);
        let r = m.mul_vec(&v);
        assert!((r.x - 1.0).abs() < f64::EPSILON);
        assert!((r.y - 2.0).abs() < f64::EPSILON);
        assert!((r.z - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mul_mat_identity() {
        let m = Matrix3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let id = Matrix3::identity();
        let r = m.mul_mat(&id);
        for i in 0..9 {
            assert!((r.data[i] - m.data[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_from_quaternion_identity() {
        let q = Quaternion::identity();
        let m = Matrix3::from_quaternion(&q);
        for i in 0..9 {
            let expected = Matrix3::identity().data[i];
            assert!(
                (m.data[i] - expected).abs() < 1e-12,
                "mismatch at index {i}"
            );
        }
    }

    #[test]
    fn test_from_quaternion_rotation_det_is_one() {
        let axis = Vector3::new(1.0, 1.0, 1.0);
        let q = Quaternion::from_axis_angle(&axis, 1.0).unwrap();
        let m = Matrix3::from_quaternion(&q);
        assert!((m.determinant() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_quaternion_is_orthogonal() {
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let q = Quaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_4).unwrap();
        let m = Matrix3::from_quaternion(&q);
        let mt = m.transpose();
        let product = m.mul_mat(&mt);
        let id = Matrix3::identity();
        for i in 0..9 {
            assert!(
                (product.data[i] - id.data[i]).abs() < 1e-10,
                "Not orthogonal at index {i}"
            );
        }
    }

    #[test]
    fn test_quaternion_rotation_matches_matrix_rotation() {
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let angle = std::f64::consts::FRAC_PI_2;
        let q = Quaternion::from_axis_angle(&axis, angle).unwrap();
        let m = Matrix3::from_quaternion(&q);

        let v = Vector3::new(1.0, 0.0, 0.0);
        let r_quat = q.rotate_vector(&v);
        let r_mat = m.mul_vec(&v);

        assert!((r_quat.x - r_mat.x).abs() < 1e-10);
        assert!((r_quat.y - r_mat.y).abs() < 1e-10);
        assert!((r_quat.z - r_mat.z).abs() < 1e-10);
    }

    #[test]
    fn test_scale() {
        let m = Matrix3::identity();
        let s = m.scale(2.0);
        assert!((s.at(0, 0) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add() {
        let a = Matrix3::identity();
        let b = Matrix3::identity();
        let c = a.add(&b);
        assert!((c.at(0, 0) - 2.0).abs() < f64::EPSILON);
        assert!(c.at(0, 1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_serde_roundtrip() {
        let m = Matrix3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let json = serde_json::to_string(&m).unwrap();
        let m2: Matrix3 = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }
}
