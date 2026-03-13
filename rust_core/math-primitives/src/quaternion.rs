//! Quaternion type and operations.
//!
//! Convention: [w, x, y, z] where w is the scalar part.

use nalgebra::Vector4;

/// Unit quaternion type (w, x, y, z).
pub type Quaternion = Vector4<f64>;

/// Create a quaternion from components.
#[inline]
pub fn quat(w: f64, x: f64, y: f64, z: f64) -> Quaternion {
    Quaternion::new(w, x, y, z)
}

/// Identity quaternion [1, 0, 0, 0].
#[inline]
pub fn identity() -> Quaternion {
    quat(1.0, 0.0, 0.0, 0.0)
}

/// Normalize a quaternion to unit length.
///
/// # Panics
/// Panics (in debug) if the quaternion has zero norm.
pub fn quaternion_normalize(q: &Quaternion) -> Quaternion {
    let norm = q.norm();
    debug_assert!(norm > 1e-15, "Cannot normalize zero quaternion");
    q / norm
}

/// Hamilton product of two quaternions.
///
/// q1 * q2 composes the rotations (q2 applied first, then q1).
pub fn quaternion_multiply(q1: &Quaternion, q2: &Quaternion) -> Quaternion {
    let (w1, x1, y1, z1) = (q1[0], q1[1], q1[2], q1[3]);
    let (w2, x2, y2, z2) = (q2[0], q2[1], q2[2], q2[3]);

    quat(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )
}

/// Quaternion inverse (conjugate / norm²).
///
/// For unit quaternions, this is just the conjugate.
pub fn quaternion_inverse(q: &Quaternion) -> Quaternion {
    let norm_sq = q.dot(q);
    debug_assert!(norm_sq > 1e-15, "Cannot invert zero quaternion");
    quat(q[0], -q[1], -q[2], -q[3]) / norm_sq
}

/// Spherical linear interpolation between two quaternions.
///
/// # Arguments
/// * `q1` - Start quaternion
/// * `q2` - End quaternion
/// * `t` - Interpolation parameter in [0, 1]
///
/// # Preconditions
/// * `t` must be in [0, 1]
/// * Both quaternions should be unit quaternions
pub fn slerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Quaternion {
    debug_assert!((0.0..=1.0).contains(&t), "t must be in [0, 1], got {t}");

    let mut q2_adj = *q2;
    let mut dot = q1.dot(q2);

    // Handle antipodal quaternions (shortest path)
    if dot < 0.0 {
        q2_adj = -q2_adj;
        dot = -dot;
    }

    // Near-linear interpolation for very close quaternions
    if dot > 0.9995 {
        let result = q1 + t * (q2_adj - q1);
        return quaternion_normalize(&result);
    }

    let theta_0 = dot.acos();
    let theta = theta_0 * t;
    let sin_theta = theta.sin();
    let sin_theta_0 = theta_0.sin();

    let s1 = theta.cos() - dot * sin_theta / sin_theta_0;
    let s2 = sin_theta / sin_theta_0;

    s1 * q1 + s2 * q2_adj
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn quat_approx_eq(a: &Quaternion, b: &Quaternion, tol: f64) -> bool {
        (a - b).norm() < tol
    }

    #[test]
    fn test_identity_quaternion() {
        let q = identity();
        assert!(approx_eq(q[0], 1.0, 1e-12));
        assert!(approx_eq(q.norm(), 1.0, 1e-12));
    }

    #[test]
    fn test_quaternion_multiply_identity() {
        let q = quat(0.5, 0.5, 0.5, 0.5);
        let id = identity();
        let result = quaternion_multiply(&q, &id);
        assert!(quat_approx_eq(&result, &q, 1e-12));
    }

    #[test]
    fn test_quaternion_multiply_inverse_gives_identity() {
        let q = quaternion_normalize(&quat(1.0, 2.0, 3.0, 4.0));
        let q_inv = quaternion_inverse(&q);
        let result = quaternion_multiply(&q, &q_inv);
        let id = identity();
        assert!(
            quat_approx_eq(&result, &id, 1e-10),
            "q * q^-1 should be identity, got {result:?}"
        );
    }

    #[test]
    fn test_quaternion_normalize() {
        let q = quat(1.0, 2.0, 3.0, 4.0);
        let qn = quaternion_normalize(&q);
        assert!(approx_eq(qn.norm(), 1.0, 1e-12));
    }

    #[test]
    fn test_slerp_endpoints() {
        let q1 = identity();
        let q2 = quaternion_normalize(&quat(0.0, 1.0, 0.0, 0.0));

        let r0 = slerp(&q1, &q2, 0.0);
        let r1 = slerp(&q1, &q2, 1.0);

        assert!(quat_approx_eq(&r0, &q1, 1e-10));
        assert!(quat_approx_eq(&r1, &q2, 1e-10));
    }

    #[test]
    fn test_slerp_midpoint_is_unit() {
        let q1 = identity();
        let q2 = quaternion_normalize(&quat(0.0, 0.0, 1.0, 0.0));
        let mid = slerp(&q1, &q2, 0.5);
        assert!(
            approx_eq(mid.norm(), 1.0, 1e-10),
            "SLERP midpoint should be unit quaternion"
        );
    }

    #[test]
    fn test_multiply_associativity() {
        let a = quaternion_normalize(&quat(1.0, 0.5, 0.3, 0.1));
        let b = quaternion_normalize(&quat(0.2, 0.8, 0.1, 0.5));
        let c = quaternion_normalize(&quat(0.3, 0.1, 0.7, 0.6));

        let ab_c = quaternion_multiply(&quaternion_multiply(&a, &b), &c);
        let a_bc = quaternion_multiply(&a, &quaternion_multiply(&b, &c));

        assert!(
            quat_approx_eq(&ab_c, &a_bc, 1e-10),
            "Quaternion multiplication should be associative"
        );
    }
}
