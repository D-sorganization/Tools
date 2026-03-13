//! Rotation representation conversions.
//!
//! Euler angles (ZYX convention), rotation matrices (SO(3)),
//! quaternions ([w,x,y,z]), and axis-angle.

use nalgebra::{Matrix3, Vector3};

use crate::quaternion::Quaternion;

/// Euler angles [roll, pitch, yaw] → 3×3 rotation matrix (ZYX convention).
///
/// R = Rz(yaw) * Ry(pitch) * Rx(roll)
pub fn euler_to_rotation_matrix(euler: &[f64; 3]) -> Matrix3<f64> {
    let (roll, pitch, yaw) = (euler[0], euler[1], euler[2]);
    let (cr, sr) = (roll.cos(), roll.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let (cy, sy) = (yaw.cos(), yaw.sin());

    Matrix3::new(
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )
}

/// 3×3 rotation matrix → Euler angles [roll, pitch, yaw] (ZYX convention).
///
/// Handles gimbal lock at pitch = ±π/2.
pub fn rotation_matrix_to_euler(r: &Matrix3<f64>) -> [f64; 3] {
    if r[(2, 0)].abs() >= 1.0 - 1e-10 {
        // Gimbal lock
        let yaw = 0.0;
        let (pitch, roll) = if r[(2, 0)] < 0.0 {
            (std::f64::consts::FRAC_PI_2, r[(0, 1)].atan2(r[(0, 2)]))
        } else {
            (-std::f64::consts::FRAC_PI_2, (-r[(0, 1)]).atan2(-r[(0, 2)]))
        };
        [roll, pitch, yaw]
    } else {
        let pitch = -r[(2, 0)].asin();
        let cp = pitch.cos();
        let roll = (r[(2, 1)] / cp).atan2(r[(2, 2)] / cp);
        let yaw = (r[(1, 0)] / cp).atan2(r[(0, 0)] / cp);
        [roll, pitch, yaw]
    }
}

/// Euler angles [roll, pitch, yaw] → quaternion [w, x, y, z].
pub fn euler_to_quaternion(euler: &[f64; 3]) -> Quaternion {
    let (hr, hp, hy) = (euler[0] / 2.0, euler[1] / 2.0, euler[2] / 2.0);
    let (cr, sr) = (hr.cos(), hr.sin());
    let (cp, sp) = (hp.cos(), hp.sin());
    let (cy, sy) = (hy.cos(), hy.sin());

    Quaternion {
        w: cr * cp * cy + sr * sp * sy,
        x: sr * cp * cy - cr * sp * sy,
        y: cr * sp * cy + sr * cp * sy,
        z: cr * cp * sy - sr * sp * cy,
    }
}

/// Quaternion [w, x, y, z] → Euler angles [roll, pitch, yaw].
pub fn quaternion_to_euler(q: &Quaternion) -> [f64; 3] {
    let (w, x, y, z) = (q.w, q.x, q.y, q.z);

    // Roll (x-axis)
    let sinr_cosp = 2.0 * (w * x + y * z);
    let cosr_cosp = 1.0 - 2.0 * (x * x + y * y);
    let roll = sinr_cosp.atan2(cosr_cosp);

    // Pitch (y-axis) — handle gimbal lock
    let sinp = 2.0 * (w * y - z * x);
    let pitch = if sinp.abs() >= 1.0 {
        std::f64::consts::FRAC_PI_2.copysign(sinp)
    } else {
        sinp.asin()
    };

    // Yaw (z-axis)
    let siny_cosp = 2.0 * (w * z + x * y);
    let cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
    let yaw = siny_cosp.atan2(cosy_cosp);

    [roll, pitch, yaw]
}

/// Quaternion [w, x, y, z] → 3×3 rotation matrix.
pub fn quaternion_to_rotation_matrix(q: &Quaternion) -> Matrix3<f64> {
    let qn_mag = q.magnitude();
    let (w, x, y, z) = (q.w / qn_mag, q.x / qn_mag, q.y / qn_mag, q.z / qn_mag);

    Matrix3::new(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
}

/// 3×3 rotation matrix → quaternion [w, x, y, z].
///
/// Uses Shepperd's method for numerical stability.
pub fn rotation_matrix_to_quaternion(r: &Matrix3<f64>) -> Quaternion {
    let trace = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];

    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        Quaternion {
            w: 0.25 / s,
            x: (r[(2, 1)] - r[(1, 2)]) * s,
            y: (r[(0, 2)] - r[(2, 0)]) * s,
            z: (r[(1, 0)] - r[(0, 1)]) * s,
        }
    } else if r[(0, 0)] > r[(1, 1)] && r[(0, 0)] > r[(2, 2)] {
        let s = 2.0 * (1.0 + r[(0, 0)] - r[(1, 1)] - r[(2, 2)]).sqrt();
        Quaternion {
            w: (r[(2, 1)] - r[(1, 2)]) / s,
            x: 0.25 * s,
            y: (r[(0, 1)] + r[(1, 0)]) / s,
            z: (r[(0, 2)] + r[(2, 0)]) / s,
        }
    } else if r[(1, 1)] > r[(2, 2)] {
        let s = 2.0 * (1.0 + r[(1, 1)] - r[(0, 0)] - r[(2, 2)]).sqrt();
        Quaternion {
            w: (r[(0, 2)] - r[(2, 0)]) / s,
            x: (r[(0, 1)] + r[(1, 0)]) / s,
            y: 0.25 * s,
            z: (r[(1, 2)] + r[(2, 1)]) / s,
        }
    } else {
        let s = 2.0 * (1.0 + r[(2, 2)] - r[(0, 0)] - r[(1, 1)]).sqrt();
        Quaternion {
            w: (r[(1, 0)] - r[(0, 1)]) / s,
            x: (r[(0, 2)] + r[(2, 0)]) / s,
            y: (r[(1, 2)] + r[(2, 1)]) / s,
            z: 0.25 * s,
        }
    }
}

/// Axis-angle → 3×3 rotation matrix (Rodrigues formula).
///
/// # Arguments
/// * `axis` - Unit rotation axis
/// * `angle` - Rotation angle in radians
pub fn axis_angle_to_rotation_matrix(axis: &[f64; 3], angle: f64) -> Matrix3<f64> {
    let a = Vector3::new(axis[0], axis[1], axis[2]);
    let a = a / a.norm(); // normalize

    let k = Matrix3::new(0.0, -a[2], a[1], a[2], 0.0, -a[0], -a[1], a[0], 0.0);

    Matrix3::identity() + angle.sin() * k + (1.0 - angle.cos()) * (k * k)
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

    #[test]
    fn test_euler_to_rotation_identity() {
        let r = euler_to_rotation_matrix(&[0.0, 0.0, 0.0]);
        let id = Matrix3::identity();
        assert!((r - id).norm() < 1e-12);
    }

    #[test]
    fn test_euler_roundtrip() {
        let euler = [0.3, 0.5, 0.7];
        let r = euler_to_rotation_matrix(&euler);
        let euler2 = rotation_matrix_to_euler(&r);
        for i in 0..3 {
            assert!(
                approx_eq(euler[i], euler2[i], 1e-10),
                "Euler roundtrip failed at index {i}: {:.6} vs {:.6}",
                euler[i],
                euler2[i]
            );
        }
    }

    #[test]
    fn test_quaternion_roundtrip_via_euler() {
        let euler = [0.2, -0.3, 1.1];
        let q = euler_to_quaternion(&euler);
        let euler2 = quaternion_to_euler(&q);
        for i in 0..3 {
            assert!(
                approx_eq(euler[i], euler2[i], 1e-10),
                "Quaternion-euler roundtrip failed at {i}"
            );
        }
    }

    #[test]
    fn test_rotation_matrix_to_quaternion_roundtrip() {
        let euler = [0.5, -0.2, 0.8];
        let r = euler_to_rotation_matrix(&euler);
        let q = rotation_matrix_to_quaternion(&r);
        let r2 = quaternion_to_rotation_matrix(&q);
        assert!((r - r2).norm() < 1e-10);
    }

    #[test]
    fn test_rotation_matrix_is_so3() {
        let euler = [1.0, -0.5, 2.0];
        let r = euler_to_rotation_matrix(&euler);
        // R^T * R should be identity
        let rtr = r.transpose() * r;
        assert!((rtr - Matrix3::identity()).norm() < 1e-10);
        // det should be +1
        assert!(approx_eq(r.determinant(), 1.0, 1e-10));
    }

    #[test]
    fn test_axis_angle_180_degrees() {
        let r = axis_angle_to_rotation_matrix(&[0.0, 0.0, 1.0], std::f64::consts::PI);
        // 180° about z: x→-x, y→-y, z→z
        assert!(approx_eq(r[(0, 0)], -1.0, 1e-10));
        assert!(approx_eq(r[(1, 1)], -1.0, 1e-10));
        assert!(approx_eq(r[(2, 2)], 1.0, 1e-10));
    }

    #[test]
    fn test_axis_angle_90_about_x() {
        let r = axis_angle_to_rotation_matrix(&[1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2);
        // 90° about x: y→z, z→-y
        assert!(approx_eq(r[(1, 1)], 0.0, 1e-10)); // cos(90)
        assert!(approx_eq(r[(2, 1)], 1.0, 1e-10)); // sin(90)
    }

    #[test]
    fn test_gimbal_lock_handling() {
        // Pitch = π/2 (gimbal lock)
        let euler = [0.0, std::f64::consts::FRAC_PI_2, 0.0];
        let r = euler_to_rotation_matrix(&euler);
        let euler2 = rotation_matrix_to_euler(&r);
        // At gimbal lock, pitch should still be recovered
        assert!(approx_eq(euler2[1], std::f64::consts::FRAC_PI_2, 1e-8));
    }
}
