//! Core mathematical types used throughout the simulation kernel.
//!
//! These are the **single source of truth** for spatial representations.
//! Python and WASM consumers receive these exact types via bindings.

use serde::{Deserialize, Serialize};

// ── Vector3 ──────────────────────────────────────────────────────────────────

/// A 3-dimensional vector used for positions, velocities, and forces.
///
/// # Design by Contract
/// - No NaN values are permitted in public constructors.
/// - `magnitude()` returns non-negative values.
/// - `normalized()` returns a unit vector or an error for zero-length vectors.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyo3::prelude::pyclass)]
#[cfg_attr(feature = "wasm", wasm_bindgen::prelude::wasm_bindgen)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    /// Create a new Vector3.
    ///
    /// # Contracts (DbC)
    /// - Precondition: No component is NaN.
    #[must_use]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        debug_assert!(!x.is_nan(), "Vector3::new: x must not be NaN");
        debug_assert!(!y.is_nan(), "Vector3::new: y must not be NaN");
        debug_assert!(!z.is_nan(), "Vector3::new: z must not be NaN");
        Self { x, y, z }
    }

    /// The zero vector.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Squared magnitude (avoids a sqrt; useful for comparisons).
    #[must_use]
    pub fn magnitude_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Euclidean magnitude.
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    /// Dot product.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product.
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Return a unit-length vector in the same direction.
    ///
    /// # Errors
    /// Returns `Err` if the vector has zero magnitude (DbC: cannot normalize zero).
    pub fn normalized(&self) -> Result<Self, &'static str> {
        let mag = self.magnitude();
        if mag < f64::EPSILON {
            return Err("Cannot normalize a zero-length vector");
        }
        Ok(Self::new(self.x / mag, self.y / mag, self.z / mag))
    }

    /// Component-wise addition.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Component-wise subtraction.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    /// Scalar multiplication.
    #[must_use]
    pub fn scale(&self, s: f64) -> Self {
        debug_assert!(!s.is_nan(), "Vector3::scale: scalar must not be NaN");
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::fmt::Display for Vector3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:.6}, {:.6}, {:.6})", self.x, self.y, self.z)
    }
}

// ── Operator Overloading (idiomatic Rust arithmetic) ─────────────────────────

impl std::ops::Add for Vector3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl std::ops::Sub for Vector3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl std::ops::Neg for Vector3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Scalar multiplication: `vector * scalar`.
impl std::ops::Mul<f64> for Vector3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

/// Scalar multiplication: `scalar * vector`.
impl std::ops::Mul<Vector3> for f64 {
    type Output = Vector3;

    fn mul(self, rhs: Vector3) -> Self::Output {
        rhs.scale(self)
    }
}

// ── Python methods (feature-gated) ───────────────────────────────────────────

#[cfg(feature = "python")]
#[pyo3::prelude::pymethods]
impl Vector3 {
    /// Create a 3D vector with components (x, y, z).
    #[new]
    #[pyo3(text_signature = "(x, y, z)")]
    fn py_new(x: f64, y: f64, z: f64) -> Self {
        Self::new(x, y, z)
    }

    /// X component of the vector.
    #[getter]
    fn x(&self) -> f64 {
        self.x
    }

    /// Y component of the vector.
    #[getter]
    fn y(&self) -> f64 {
        self.y
    }

    /// Z component of the vector.
    #[getter]
    fn z(&self) -> f64 {
        self.z
    }

    fn __repr__(&self) -> String {
        format!("Vector3({}, {}, {})", self.x, self.y, self.z)
    }

    fn __str__(&self) -> String {
        format!("{self}")
    }

    /// Return the Euclidean magnitude (length) of the vector.
    #[pyo3(name = "magnitude", text_signature = "($self)")]
    fn py_magnitude(&self) -> f64 {
        self.magnitude()
    }

    /// Compute the dot product with another Vector3.
    #[pyo3(name = "dot", text_signature = "($self, other)")]
    fn py_dot(&self, other: &Self) -> f64 {
        self.dot(other)
    }

    /// Compute the cross product with another Vector3.
    #[pyo3(name = "cross", text_signature = "($self, other)")]
    fn py_cross(&self, other: &Self) -> Self {
        self.cross(other)
    }

    /// Return a unit vector in the same direction.
    ///
    /// Raises ValueError if the vector has zero magnitude.
    #[pyo3(name = "normalized", text_signature = "($self)")]
    fn py_normalized(&self) -> pyo3::PyResult<Self> {
        self.normalized()
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Return a new vector scaled by the given factor.
    #[pyo3(name = "scale", text_signature = "($self, s)")]
    fn py_scale(&self, s: f64) -> Self {
        self.scale(s)
    }
}

// ── Tests (TDD — written before implementation was finalized) ────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ──

    #[test]
    fn test_new_creates_vector() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_zero_vector() {
        let v = Vector3::zero();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    // ── DbC: NaN rejection ──

    #[test]
    #[should_panic(expected = "x must not be NaN")]
    fn test_nan_x_panics_in_debug() {
        let _v = Vector3::new(f64::NAN, 0.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "y must not be NaN")]
    fn test_nan_y_panics_in_debug() {
        let _v = Vector3::new(0.0, f64::NAN, 0.0);
    }

    #[test]
    #[should_panic(expected = "scalar must not be NaN")]
    fn test_scale_nan_panics_in_debug() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let _r = v.scale(f64::NAN);
    }

    // ── Magnitude ──

    #[test]
    fn test_magnitude_of_unit_x() {
        let v = Vector3::new(1.0, 0.0, 0.0);
        assert!((v.magnitude() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_magnitude_of_3_4_0() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        assert!((v.magnitude() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_magnitude_squared() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert!((v.magnitude_squared() - 14.0).abs() < f64::EPSILON);
    }

    // ── Dot product ──

    #[test]
    fn test_dot_product_orthogonal() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 1.0, 0.0);
        assert!((a.dot(&b)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dot_product_parallel() {
        let a = Vector3::new(2.0, 0.0, 0.0);
        let b = Vector3::new(3.0, 0.0, 0.0);
        assert!((a.dot(&b) - 6.0).abs() < f64::EPSILON);
    }

    // ── Cross product ──

    #[test]
    fn test_cross_product_xy_gives_z() {
        let x = Vector3::new(1.0, 0.0, 0.0);
        let y = Vector3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!((z.x).abs() < f64::EPSILON);
        assert!((z.y).abs() < f64::EPSILON);
        assert!((z.z - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cross_product_anticommutative() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);
        let ab = a.cross(&b);
        let ba = b.cross(&a);
        assert!((ab.x + ba.x).abs() < 1e-12);
        assert!((ab.y + ba.y).abs() < 1e-12);
        assert!((ab.z + ba.z).abs() < 1e-12);
    }

    // ── Normalization ──

    #[test]
    fn test_normalized_unit_vector() {
        let v = Vector3::new(3.0, 4.0, 0.0);
        let n = v.normalized().unwrap();
        assert!((n.magnitude() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_normalized_zero_returns_error() {
        let v = Vector3::zero();
        assert!(v.normalized().is_err());
    }

    // ── Arithmetic ──

    #[test]
    fn test_add() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);
        let c = a.add(&b);
        assert_eq!(c, Vector3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_sub() {
        let a = Vector3::new(4.0, 5.0, 6.0);
        let b = Vector3::new(1.0, 2.0, 3.0);
        let c = a.sub(&b);
        assert_eq!(c, Vector3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_scale() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let s = v.scale(2.0);
        assert_eq!(s, Vector3::new(2.0, 4.0, 6.0));
    }
    // ── Operator Overloading ──

    #[test]
    fn test_add_operator() {
        let a = Vector3::new(1.0, 2.0, 3.0);
        let b = Vector3::new(4.0, 5.0, 6.0);
        let c = a + b;
        assert_eq!(c, Vector3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_sub_operator() {
        let a = Vector3::new(4.0, 5.0, 6.0);
        let b = Vector3::new(1.0, 2.0, 3.0);
        let c = a - b;
        assert_eq!(c, Vector3::new(3.0, 3.0, 3.0));
    }

    #[test]
    fn test_neg_operator() {
        let a = Vector3::new(1.0, -2.0, 3.0);
        let b = -a;
        assert_eq!(b, Vector3::new(-1.0, 2.0, -3.0));
    }

    #[test]
    fn test_mul_scalar_right() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let s = v * 2.0;
        assert_eq!(s, Vector3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_mul_scalar_left() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let s = 2.0 * v;
        assert_eq!(s, Vector3::new(2.0, 4.0, 6.0));
    }

    // ── Serialization (DRY: Serde is the canonical format) ──

    #[test]
    fn test_serde_roundtrip() {
        let v = Vector3::new(1.5, -2.5, 7.77);
        let json = serde_json::to_string(&v).unwrap();
        let v2: Vector3 = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    // ── Display ──

    #[test]
    fn test_display() {
        let v = Vector3::new(1.0, 2.0, 3.0);
        let s = format!("{v}");
        assert!(s.contains("1.000000"));
    }
}

// ── Property-Based Tests (proptest) ──────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::needless_pass_by_value, clippy::float_cmp)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate finite f64 values (no NaN/Inf) in a reasonable range.
    fn finite_f64() -> impl Strategy<Value = f64> {
        -1e6_f64..1e6_f64
    }

    /// Generate a Vector3 with finite components.
    fn arb_vector3() -> impl Strategy<Value = Vector3> {
        (finite_f64(), finite_f64(), finite_f64()).prop_map(|(x, y, z)| Vector3::new(x, y, z))
    }

    proptest! {
        /// Magnitude is always non-negative.
        #[test]
        fn magnitude_is_non_negative(v in arb_vector3()) {
            prop_assert!(v.magnitude() >= 0.0);
        }

        /// Addition is commutative: a + b == b + a.
        #[test]
        fn add_is_commutative(a in arb_vector3(), b in arb_vector3()) {
            let ab = a.add(&b);
            let ba = b.add(&a);
            prop_assert!((ab.x - ba.x).abs() < 1e-10);
            prop_assert!((ab.y - ba.y).abs() < 1e-10);
            prop_assert!((ab.z - ba.z).abs() < 1e-10);
        }

        /// Dot product is commutative: a · b == b · a.
        #[test]
        fn dot_is_commutative(a in arb_vector3(), b in arb_vector3()) {
            let ab = a.dot(&b);
            let ba = b.dot(&a);
            prop_assert!((ab - ba).abs() < 1e-6);
        }

        /// Cross product is anti-commutative: a × b == -(b × a).
        #[test]
        fn cross_is_anticommutative(a in arb_vector3(), b in arb_vector3()) {
            let ab = a.cross(&b);
            let ba = b.cross(&a);
            prop_assert!((ab.x + ba.x).abs() < 1e-6);
            prop_assert!((ab.y + ba.y).abs() < 1e-6);
            prop_assert!((ab.z + ba.z).abs() < 1e-6);
        }

        /// Normalized vector has magnitude 1 (when non-zero).
        #[test]
        fn normalized_has_unit_length(v in arb_vector3()) {
            if v.magnitude() > 1e-10 {
                let n = v.normalized().unwrap();
                prop_assert!((n.magnitude() - 1.0).abs() < 1e-10);
            }
        }

        /// Scalar multiplication distributes: s * (a + b) == s*a + s*b.
        #[test]
        fn scale_distributes_over_add(
            a in arb_vector3(),
            b in arb_vector3(),
            s in finite_f64()
        ) {
            let left = a.add(&b).scale(s);
            let right = a.scale(s).add(&b.scale(s));
            // Use relative tolerance: |diff| / max(|left|, |right|, 1) < eps
            let tol = 1e-8;
            let denom = left.magnitude().max(right.magnitude()).max(1.0);
            prop_assert!((left.x - right.x).abs() / denom < tol);
            prop_assert!((left.y - right.y).abs() / denom < tol);
            prop_assert!((left.z - right.z).abs() / denom < tol);
        }

        /// Cross product with self is zero: v × v == 0.
        #[test]
        fn cross_self_is_zero(v in arb_vector3()) {
            let c = v.cross(&v);
            let denom = v.magnitude().max(1.0);
            prop_assert!(c.magnitude() / denom < 1e-8);
        }
    }
}
