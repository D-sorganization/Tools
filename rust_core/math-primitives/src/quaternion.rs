//! Unit quaternion for rotation representations.
//!
//! The **canonical rotation type** for the simulation kernel.
//! All rotation conversions flow through this type.
//!
//! # Design by Contract
//! - Quaternions are stored normalized. `new()` normalizes automatically.
//! - `debug_assert!` rejects NaN and zero-magnitude input.

use serde::{Deserialize, Serialize};

use crate::types::Vector3;

/// A unit quaternion representing a 3D rotation.
///
/// Stored as `(w, x, y, z)` where `w` is the scalar part.
/// Always normalized (magnitude = 1) to represent valid rotations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass(from_py_object))]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    /// Create a new quaternion and normalize it.
    ///
    /// # Contracts (DbC)
    /// - Precondition: No component is NaN.
    /// - Precondition: At least one component is non-zero.
    /// - Postcondition: The returned quaternion has unit magnitude.
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Result<Self, &'static str> {
        debug_assert!(!w.is_nan(), "Quaternion::new: w must not be NaN");
        debug_assert!(!x.is_nan(), "Quaternion::new: x must not be NaN");
        debug_assert!(!y.is_nan(), "Quaternion::new: y must not be NaN");
        debug_assert!(!z.is_nan(), "Quaternion::new: z must not be NaN");

        let mag = (w * w + x * x + y * y + z * z).sqrt();
        if mag < f64::EPSILON {
            return Err("Cannot create quaternion from zero-magnitude input");
        }
        Ok(Self {
            w: w / mag,
            x: x / mag,
            y: y / mag,
            z: z / mag,
        })
    }

    /// The identity quaternion (no rotation).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Create from axis-angle representation.
    ///
    /// # Contracts (DbC)
    /// - Precondition: `axis` must be non-zero (will be normalized).
    /// - Precondition: `angle_rad` must not be NaN.
    pub fn from_axis_angle(axis: &Vector3, angle_rad: f64) -> Result<Self, &'static str> {
        debug_assert!(
            !angle_rad.is_nan(),
            "from_axis_angle: angle must not be NaN"
        );
        let n = axis.normalized()?;
        let half = angle_rad * 0.5;
        let s = half.sin();
        // axis-angle to quaternion always produces unit quaternion
        Ok(Self {
            w: half.cos(),
            x: n.x * s,
            y: n.y * s,
            z: n.z * s,
        })
    }

    /// Squared magnitude (should always be ~1.0 for unit quaternions).
    #[must_use]
    pub fn magnitude_squared(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Euclidean magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Conjugate (inverse for unit quaternions).
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Hamilton product (quaternion multiplication).
    ///
    /// # Contracts (DbC)
    /// - Postcondition: result has approximately unit magnitude.
    #[must_use]
    pub fn multiply(&self, other: &Self) -> Self {
        let result = Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        };
        // Postcondition: unit × unit = unit (skip for embedded vectors in rotate_vector)
        debug_assert!(
            (self.magnitude_squared() - 1.0).abs() > 0.1
                || (other.magnitude_squared() - 1.0).abs() > 0.1
                || (result.magnitude_squared() - 1.0).abs() < 1e-6,
            "DbC postcondition: unit quaternion product must be unit, got mag²={}",
            result.magnitude_squared()
        );
        result
    }

    /// Rotate a vector by this quaternion: v' = q * v * q⁻¹
    #[must_use]
    pub fn rotate_vector(&self, v: &Vector3) -> Vector3 {
        let q_v = Quaternion {
            w: 0.0,
            x: v.x,
            y: v.y,
            z: v.z,
        };
        let result = self.multiply(&q_v).multiply(&self.conjugate());
        Vector3::new(result.x, result.y, result.z)
    }

    /// Spherical linear interpolation between two quaternions.
    ///
    /// # Contracts (DbC)
    /// - Precondition: `t` in [0, 1] (clamped in release mode for safety).
    /// - Postcondition: result is a unit quaternion.
    #[must_use]
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        debug_assert!(
            (0.0..=1.0).contains(&t),
            "slerp: t must be in [0.0, 1.0], got {t}"
        );
        // Clamp in release mode to prevent extrapolation
        let t = t.clamp(0.0, 1.0);

        let mut dot = self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z;

        // If dot is negative, negate one quaternion to take the shorter path
        let mut other_adj = *other;
        if dot < 0.0 {
            other_adj = Quaternion {
                w: -other.w,
                x: -other.x,
                y: -other.y,
                z: -other.z,
            };
            dot = -dot;
        }

        // If quaternions are very close, use linear interpolation
        if dot > 0.9995 {
            let w = self.w + t * (other_adj.w - self.w);
            let x = self.x + t * (other_adj.x - self.x);
            let y = self.y + t * (other_adj.y - self.y);
            let z = self.z + t * (other_adj.z - self.z);
            let mag = (w * w + x * x + y * y + z * z).sqrt();
            return Quaternion {
                w: w / mag,
                x: x / mag,
                y: y / mag,
                z: z / mag,
            };
        }

        let theta = dot.acos();
        let sin_theta = theta.sin();
        let s0 = ((1.0 - t) * theta).sin() / sin_theta;
        let s1 = (t * theta).sin() / sin_theta;

        Quaternion {
            w: s0 * self.w + s1 * other_adj.w,
            x: s0 * self.x + s1 * other_adj.x,
            y: s0 * self.y + s1 * other_adj.y,
            z: s0 * self.z + s1 * other_adj.z,
        }
    }
}

impl std::fmt::Display for Quaternion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Quaternion({:.6}, {:.6}, {:.6}, {:.6})",
            self.w, self.x, self.y, self.z
        )
    }
}

// ── Python bindings ──────────────────────────────────────────────────────────

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl Quaternion {
    /// Create a new unit quaternion (automatically normalized).
    #[new]
    #[pyo3(text_signature = "(w, x, y, z)")]
    fn py_new(w: f64, x: f64, y: f64, z: f64) -> pyo3::PyResult<Self> {
        Self::new(w, x, y, z).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Scalar component.
    #[getter]
    fn w(&self) -> f64 {
        self.w
    }
    /// X imaginary component.
    #[getter]
    fn x(&self) -> f64 {
        self.x
    }
    /// Y imaginary component.
    #[getter]
    fn y(&self) -> f64 {
        self.y
    }
    /// Z imaginary component.
    #[getter]
    fn z(&self) -> f64 {
        self.z
    }

    fn __repr__(&self) -> String {
        format!("Quaternion({}, {}, {}, {})", self.w, self.x, self.y, self.z)
    }

    /// Return the conjugate (inverse for unit quaternions).
    #[pyo3(name = "conjugate", text_signature = "($self)")]
    fn py_conjugate(&self) -> Self {
        self.conjugate()
    }

    /// Hamilton product (quaternion multiplication).
    #[pyo3(name = "multiply", text_signature = "($self, other)")]
    fn py_multiply(&self, other: &Self) -> Self {
        self.multiply(other)
    }

    /// Rotate a 3D vector by this quaternion: v' = q * v * q⁻¹.
    #[pyo3(name = "rotate_vector", text_signature = "($self, v)")]
    fn py_rotate_vector(&self, v: &Vector3) -> Vector3 {
        self.rotate_vector(v)
    }

    /// Create a quaternion from axis-angle representation.
    #[staticmethod]
    #[pyo3(name = "from_axis_angle", text_signature = "(axis, angle_rad)")]
    fn py_from_axis_angle(axis: &Vector3, angle_rad: f64) -> pyo3::PyResult<Self> {
        Self::from_axis_angle(axis, angle_rad).map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

// ── Tests (TDD) ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_quaternion() {
        let q = Quaternion::identity();
        assert!((q.w - 1.0).abs() < f64::EPSILON);
        assert!(q.x.abs() < f64::EPSILON);
    }

    #[test]
    fn test_new_normalizes() {
        let q = Quaternion::new(2.0, 0.0, 0.0, 0.0).unwrap();
        assert!((q.magnitude() - 1.0).abs() < 1e-12);
        assert!((q.w - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_zero_input_returns_error() {
        assert!(Quaternion::new(0.0, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_conjugate() {
        let q = Quaternion::new(1.0, 1.0, 1.0, 1.0).unwrap();
        let c = q.conjugate();
        assert!((c.w - q.w).abs() < 1e-12);
        assert!((c.x + q.x).abs() < 1e-12);
        assert!((c.y + q.y).abs() < 1e-12);
        assert!((c.z + q.z).abs() < 1e-12);
    }

    #[test]
    fn test_multiply_identity() {
        let q = Quaternion::new(0.5, 0.5, 0.5, 0.5).unwrap();
        let id = Quaternion::identity();
        let result = q.multiply(&id);
        assert!((result.w - q.w).abs() < 1e-12);
        assert!((result.x - q.x).abs() < 1e-12);
    }

    #[test]
    fn test_multiply_by_conjugate_gives_identity() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0).unwrap();
        let result = q.multiply(&q.conjugate());
        assert!((result.w - 1.0).abs() < 1e-10);
        assert!(result.x.abs() < 1e-10);
        assert!(result.y.abs() < 1e-10);
        assert!(result.z.abs() < 1e-10);
    }

    #[test]
    fn test_from_axis_angle_90_deg_about_z() {
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let q = Quaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2).unwrap();
        // Rotate x-axis by 90° about z should give y-axis
        let v = Vector3::new(1.0, 0.0, 0.0);
        let r = q.rotate_vector(&v);
        assert!(r.x.abs() < 1e-10);
        assert!((r.y - 1.0).abs() < 1e-10);
        assert!(r.z.abs() < 1e-10);
    }

    #[test]
    fn test_rotate_identity_preserves_vector() {
        let q = Quaternion::identity();
        let v = Vector3::new(1.0, 2.0, 3.0);
        let r = q.rotate_vector(&v);
        assert!((r.x - v.x).abs() < 1e-12);
        assert!((r.y - v.y).abs() < 1e-12);
        assert!((r.z - v.z).abs() < 1e-12);
    }

    #[test]
    fn test_slerp_at_endpoints() {
        let q1 = Quaternion::identity();
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let q2 = Quaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2).unwrap();

        let s0 = q1.slerp(&q2, 0.0);
        assert!((s0.w - q1.w).abs() < 1e-10);

        let s1 = q1.slerp(&q2, 1.0);
        assert!((s1.w - q2.w).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_midpoint() {
        let q1 = Quaternion::identity();
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let q2 = Quaternion::from_axis_angle(&axis, std::f64::consts::FRAC_PI_2).unwrap();
        let mid = q1.slerp(&q2, 0.5);
        // Midpoint of identity and 90° rotation should be 45° rotation
        assert!((mid.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_serde_roundtrip() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0).unwrap();
        let json = serde_json::to_string(&q).unwrap();
        let q2: Quaternion = serde_json::from_str(&json).unwrap();
        assert!((q.w - q2.w).abs() < 1e-12);
    }
}
