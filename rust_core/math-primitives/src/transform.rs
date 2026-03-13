//! Pose6DOF and Transform6DOF structs.
//!
//! Pose6DOF: position + euler angles representation.
//! Transform6DOF: rotation matrix + translation representation.
//!
//! Ported from UpstreamDrift/pose6dof.py.

use nalgebra::{Matrix3, Matrix4, Vector3};

use crate::quaternion::Quaternion;
use crate::rotation::{
    euler_to_quaternion, euler_to_rotation_matrix, quaternion_to_euler,
    quaternion_to_rotation_matrix, rotation_matrix_to_euler, rotation_matrix_to_quaternion,
};

/// 6DOF Pose: position [x,y,z] + orientation [roll, pitch, yaw].
///
/// Uses ZYX Euler angle convention (yaw-pitch-roll).
#[derive(Debug, Clone)]
pub struct Pose6DOF {
    pub position: Vector3<f64>,
    pub euler_angles: [f64; 3],
}

impl Default for Pose6DOF {
    fn default() -> Self {
        Self {
            position: Vector3::zeros(),
            euler_angles: [0.0; 3],
        }
    }
}

impl Pose6DOF {
    /// Create a new pose from position and euler angles.
    pub fn new(position: [f64; 3], euler_angles: [f64; 3]) -> Self {
        Self {
            position: Vector3::new(position[0], position[1], position[2]),
            euler_angles,
        }
    }

    /// Create from position and quaternion [w, x, y, z].
    pub fn from_quaternion(position: [f64; 3], quaternion: &Quaternion) -> Self {
        let euler = quaternion_to_euler(quaternion);
        Self::new(position, euler)
    }

    /// Create from position and rotation matrix.
    pub fn from_rotation_matrix(position: [f64; 3], rotation: &Matrix3<f64>) -> Self {
        let euler = rotation_matrix_to_euler(rotation);
        Self::new(position, euler)
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    pub fn x(&self) -> f64 {
        self.position[0]
    }

    pub fn y(&self) -> f64 {
        self.position[1]
    }

    pub fn z(&self) -> f64 {
        self.position[2]
    }

    pub fn roll(&self) -> f64 {
        self.euler_angles[0]
    }

    pub fn pitch(&self) -> f64 {
        self.euler_angles[1]
    }

    pub fn yaw(&self) -> f64 {
        self.euler_angles[2]
    }

    /// Get 3×3 rotation matrix.
    pub fn rotation_matrix(&self) -> Matrix3<f64> {
        euler_to_rotation_matrix(&self.euler_angles)
    }

    /// Get 4×4 homogeneous transformation matrix.
    pub fn homogeneous_matrix(&self) -> Matrix4<f64> {
        let r = self.rotation_matrix();
        let mut h = Matrix4::identity();
        h.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        h[(0, 3)] = self.position[0];
        h[(1, 3)] = self.position[1];
        h[(2, 3)] = self.position[2];
        h
    }

    /// Convert orientation to quaternion [w, x, y, z].
    pub fn to_quaternion(&self) -> Quaternion {
        euler_to_quaternion(&self.euler_angles)
    }

    // -----------------------------------------------------------------------
    // Transformations (return new poses, immutable)
    // -----------------------------------------------------------------------

    /// Translate by offset in world frame.
    pub fn translate(&self, offset: &[f64; 3]) -> Pose6DOF {
        let off = Vector3::new(offset[0], offset[1], offset[2]);
        Pose6DOF {
            position: self.position + off,
            euler_angles: self.euler_angles,
        }
    }

    /// Apply additional euler rotation.
    pub fn rotate_euler(&self, delta_euler: &[f64; 3]) -> Pose6DOF {
        let q1 = self.to_quaternion();
        let q2 = euler_to_quaternion(delta_euler);
        let q3 = q1.multiply(&q2);
        let new_euler = quaternion_to_euler(&q3);
        Pose6DOF {
            position: self.position,
            euler_angles: new_euler,
        }
    }

    /// Inverse pose.
    pub fn inverse(&self) -> Pose6DOF {
        let r = self.rotation_matrix();
        let r_inv = r.transpose();
        let p_inv = -r_inv * self.position;
        let euler_inv = rotation_matrix_to_euler(&r_inv);
        Pose6DOF {
            position: p_inv,
            euler_angles: euler_inv,
        }
    }

    /// Compose this pose with another: self * other.
    pub fn compose(&self, other: &Pose6DOF) -> Pose6DOF {
        let r1 = self.rotation_matrix();
        let r2 = other.rotation_matrix();
        let r = r1 * r2;
        let p = r1 * other.position + self.position;
        Pose6DOF::from_rotation_matrix([p[0], p[1], p[2]], &r)
    }

    /// Transform a point by this pose.
    pub fn transform_point(&self, point: &[f64; 3]) -> [f64; 3] {
        let pt = Vector3::new(point[0], point[1], point[2]);
        let result = self.rotation_matrix() * pt + self.position;
        [result[0], result[1], result[2]]
    }

    /// Transform a direction vector (rotation only).
    pub fn transform_vector(&self, vector: &[f64; 3]) -> [f64; 3] {
        let v = Vector3::new(vector[0], vector[1], vector[2]);
        let result = self.rotation_matrix() * v;
        [result[0], result[1], result[2]]
    }
}

/// Rigid body transformation: rotation matrix + translation.
///
/// More efficient for repeated composition than Pose6DOF
/// (avoids euler↔matrix roundtrips).
#[derive(Debug, Clone)]
pub struct Transform6DOF {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

impl Default for Transform6DOF {
    fn default() -> Self {
        Self {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }
}

impl Transform6DOF {
    /// Create from rotation matrix and translation.
    pub fn new(rotation: Matrix3<f64>, translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation: Vector3::new(translation[0], translation[1], translation[2]),
        }
    }

    /// Identity transform.
    pub fn identity() -> Self {
        Self::default()
    }

    /// Pure translation.
    pub fn from_translation(t: [f64; 3]) -> Self {
        Self {
            rotation: Matrix3::identity(),
            translation: Vector3::new(t[0], t[1], t[2]),
        }
    }

    /// Rotation about X axis.
    pub fn from_rotation_x(angle: f64) -> Self {
        let (c, s) = (angle.cos(), angle.sin());
        let r = Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
        Self {
            rotation: r,
            translation: Vector3::zeros(),
        }
    }

    /// Rotation about Y axis.
    pub fn from_rotation_y(angle: f64) -> Self {
        let (c, s) = (angle.cos(), angle.sin());
        let r = Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
        Self {
            rotation: r,
            translation: Vector3::zeros(),
        }
    }

    /// Rotation about Z axis.
    pub fn from_rotation_z(angle: f64) -> Self {
        let (c, s) = (angle.cos(), angle.sin());
        let r = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
        Self {
            rotation: r,
            translation: Vector3::zeros(),
        }
    }

    /// From 4×4 homogeneous matrix.
    pub fn from_homogeneous(h: &Matrix4<f64>) -> Self {
        let r = h.fixed_view::<3, 3>(0, 0).into_owned();
        let t = [h[(0, 3)], h[(1, 3)], h[(2, 3)]];
        Self::new(r, t)
    }

    /// From Pose6DOF.
    pub fn from_pose(pose: &Pose6DOF) -> Self {
        Self {
            rotation: pose.rotation_matrix(),
            translation: pose.position,
        }
    }

    /// Get 4×4 homogeneous matrix.
    pub fn homogeneous_matrix(&self) -> Matrix4<f64> {
        let mut h = Matrix4::identity();
        h.fixed_view_mut::<3, 3>(0, 0).copy_from(&self.rotation);
        h[(0, 3)] = self.translation[0];
        h[(1, 3)] = self.translation[1];
        h[(2, 3)] = self.translation[2];
        h
    }

    /// Compose: other * self (world-frame chaining).
    pub fn compose(&self, other: &Transform6DOF) -> Transform6DOF {
        Transform6DOF {
            rotation: other.rotation * self.rotation,
            translation: other.rotation * self.translation + other.translation,
        }
    }

    /// Inverse transform.
    pub fn inverse(&self) -> Transform6DOF {
        let r_inv = self.rotation.transpose();
        let t_inv = -r_inv * self.translation;
        Transform6DOF {
            rotation: r_inv,
            translation: t_inv,
        }
    }

    /// Transform a single point.
    pub fn transform_point(&self, point: &[f64; 3]) -> [f64; 3] {
        let pt = Vector3::new(point[0], point[1], point[2]);
        let result = self.rotation * pt + self.translation;
        [result[0], result[1], result[2]]
    }

    /// Transform a direction vector (rotation only).
    pub fn transform_vector(&self, vector: &[f64; 3]) -> [f64; 3] {
        let v = Vector3::new(vector[0], vector[1], vector[2]);
        let result = self.rotation * v;
        [result[0], result[1], result[2]]
    }

    /// Convert to Pose6DOF.
    pub fn to_pose(&self) -> Pose6DOF {
        let euler = rotation_matrix_to_euler(&self.rotation);
        Pose6DOF {
            position: self.translation,
            euler_angles: euler,
        }
    }

    /// Interpolate between two transforms (SLERP + linear).
    pub fn interpolate(t1: &Transform6DOF, t2: &Transform6DOF, alpha: f64) -> Transform6DOF {
        debug_assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0, 1]");

        let translation = (1.0 - alpha) * t1.translation + alpha * t2.translation;

        let q1 = rotation_matrix_to_quaternion(&t1.rotation);
        let q2 = rotation_matrix_to_quaternion(&t2.rotation);
        let q = q1.slerp(&q2, alpha);
        let rotation = quaternion_to_rotation_matrix(&q);

        Transform6DOF {
            rotation,
            translation,
        }
    }
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
    fn test_pose_default_is_origin() {
        let p = Pose6DOF::default();
        assert!(approx_eq(p.x(), 0.0, 1e-12));
        assert!(approx_eq(p.y(), 0.0, 1e-12));
        assert!(approx_eq(p.z(), 0.0, 1e-12));
        assert!(approx_eq(p.roll(), 0.0, 1e-12));
        assert!(approx_eq(p.pitch(), 0.0, 1e-12));
        assert!(approx_eq(p.yaw(), 0.0, 1e-12));
    }

    #[test]
    fn test_pose_translate() {
        let p = Pose6DOF::new([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]);
        let p2 = p.translate(&[10.0, 20.0, 30.0]);
        assert!(approx_eq(p2.x(), 11.0, 1e-12));
        assert!(approx_eq(p2.y(), 22.0, 1e-12));
        assert!(approx_eq(p2.z(), 33.0, 1e-12));
    }

    #[test]
    fn test_pose_inverse_compose_is_identity() {
        let p = Pose6DOF::new([1.0, 2.0, 3.0], [0.3, -0.2, 0.5]);
        let p_inv = p.inverse();
        let result = p.compose(&p_inv);
        assert!(approx_eq(result.x(), 0.0, 1e-8));
        assert!(approx_eq(result.y(), 0.0, 1e-8));
        assert!(approx_eq(result.z(), 0.0, 1e-8));
    }

    #[test]
    fn test_pose_transform_point() {
        let p = Pose6DOF::new([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let pt = p.transform_point(&[0.0, 0.0, 0.0]);
        assert!(approx_eq(pt[0], 1.0, 1e-12));
    }

    #[test]
    fn test_pose_homogeneous_roundtrip() {
        let p = Pose6DOF::new([1.0, 2.0, 3.0], [0.1, -0.2, 0.3]);
        let h = p.homogeneous_matrix();
        let t = Transform6DOF::from_homogeneous(&h);
        let p2 = t.to_pose();
        assert!(approx_eq(p.x(), p2.x(), 1e-10));
        assert!(approx_eq(p.y(), p2.y(), 1e-10));
        assert!(approx_eq(p.z(), p2.z(), 1e-10));
    }

    #[test]
    fn test_transform_identity() {
        let t = Transform6DOF::identity();
        let pt = t.transform_point(&[1.0, 2.0, 3.0]);
        assert!(approx_eq(pt[0], 1.0, 1e-12));
        assert!(approx_eq(pt[1], 2.0, 1e-12));
        assert!(approx_eq(pt[2], 3.0, 1e-12));
    }

    #[test]
    fn test_transform_compose_inverse_is_identity() {
        let t = Transform6DOF::new(euler_to_rotation_matrix(&[0.3, 0.5, -0.2]), [1.0, 2.0, 3.0]);
        let t_inv = t.inverse();
        let result = t.compose(&t_inv);
        let pt = result.transform_point(&[5.0, 6.0, 7.0]);
        assert!(approx_eq(pt[0], 5.0, 1e-8));
        assert!(approx_eq(pt[1], 6.0, 1e-8));
        assert!(approx_eq(pt[2], 7.0, 1e-8));
    }

    #[test]
    fn test_transform_rotation_x() {
        let t = Transform6DOF::from_rotation_x(std::f64::consts::FRAC_PI_2);
        let pt = t.transform_point(&[0.0, 1.0, 0.0]);
        assert!(approx_eq(pt[0], 0.0, 1e-10));
        assert!(approx_eq(pt[1], 0.0, 1e-10));
        assert!(approx_eq(pt[2], 1.0, 1e-10));
    }

    #[test]
    fn test_transform_interpolate_endpoints() {
        let t1 = Transform6DOF::from_translation([0.0, 0.0, 0.0]);
        let t2 = Transform6DOF::from_translation([10.0, 20.0, 30.0]);

        let r0 = Transform6DOF::interpolate(&t1, &t2, 0.0);
        assert!(approx_eq(r0.translation[0], 0.0, 1e-10));

        let r1 = Transform6DOF::interpolate(&t1, &t2, 1.0);
        assert!(approx_eq(r1.translation[0], 10.0, 1e-10));
        assert!(approx_eq(r1.translation[1], 20.0, 1e-10));

        let r_mid = Transform6DOF::interpolate(&t1, &t2, 0.5);
        assert!(approx_eq(r_mid.translation[0], 5.0, 1e-10));
    }

    #[test]
    fn test_from_pose_roundtrip() {
        let pose = Pose6DOF::new([1.0, 2.0, 3.0], [0.1, -0.2, 0.3]);
        let t = Transform6DOF::from_pose(&pose);
        let pose2 = t.to_pose();
        assert!(approx_eq(pose.x(), pose2.x(), 1e-10));
        assert!(approx_eq(pose.y(), pose2.y(), 1e-10));
        assert!(approx_eq(pose.z(), pose2.z(), 1e-10));
    }
}
