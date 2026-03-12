//! Triple pendulum (3-DOF) physics implementation.
//!
//! Model:
//! - q = [θ₁, φ₁, φ₂] where θ₁ is shoulder angle, φ₁ is elbow relative, φ₂ is wrist relative
//! - Three segments with masses m₁, m₂, m₃ and lengths l₁, l₂, l₃

use crate::types::{TripleFKResult, TriplePendulumParams};
use nalgebra::{SMatrix, SVector};

/// Compute the 3x3 mass matrix M(q).
///
/// For the desktop triple-pendulum model, the mass matrix is:
/// M[0,0] = (m1+m2+m3)l1² + (m2+m3)l2² + m3l3²
///        + 2(m2+m3)l1l2cos(φ₁) + 2m3l1l3cos(φ₁+φ₂) + 2m3l2l3cos(φ₂)
/// M[0,1] = (m2+m3)l2² + m3l3²
///        + (m2+m3)l1l2cos(φ₁) + m3l1l3cos(φ₁+φ₂) + 2m3l2l3cos(φ₂)
/// M[0,2] = m3l3² + m3l1l3cos(φ₁+φ₂) + m3l2l3cos(φ₂)
/// M[1,1] = (m2+m3)l2² + m3l3² + 2m3l2l3cos(φ₂)
/// M[1,2] = m3l3² + m3l2l3cos(φ₂)
/// M[2,2] = m3l3²
pub fn mass_matrix(q: &[f64; 3], params: &TriplePendulumParams) -> SMatrix<f64, 3, 3> {
    let phi1 = q[1];
    let phi2 = q[2];
    let sum_phi12 = phi1 + phi2;

    let cos_phi1 = phi1.cos();
    let cos_phi2 = phi2.cos();
    let cos_sum_phi12 = sum_phi12.cos();

    let l1_sq = params.lengths[0] * params.lengths[0];
    let l2_sq = params.lengths[1] * params.lengths[1];
    let l3_sq = params.lengths[2] * params.lengths[2];

    let m1 = params.masses[0];
    let m2 = params.masses[1];
    let m3 = params.masses[2];
    let m23 = m2 + m3;
    let l1 = params.lengths[0];
    let l2 = params.lengths[1];
    let l3 = params.lengths[2];

    let m00 = (m1 + m2 + m3) * l1_sq
        + m23 * l2_sq
        + m3 * l3_sq
        + 2.0 * m23 * l1 * l2 * cos_phi1
        + 2.0 * m3 * l1 * l3 * cos_sum_phi12
        + 2.0 * m3 * l2 * l3 * cos_phi2;
    let m01 = m23 * l2_sq
        + m3 * l3_sq
        + m23 * l1 * l2 * cos_phi1
        + m3 * l1 * l3 * cos_sum_phi12
        + 2.0 * m3 * l2 * l3 * cos_phi2;
    let m02 = m3 * l3_sq + m3 * l1 * l3 * cos_sum_phi12 + m3 * l2 * l3 * cos_phi2;
    let m11 = m23 * l2_sq + m3 * l3_sq + 2.0 * m3 * l2 * l3 * cos_phi2;
    let m12 = m3 * l3_sq + m3 * l2 * l3 * cos_phi2;
    let m22 = params.masses[2] * l3_sq;

    SMatrix::<f64, 3, 3>::new(m00, m01, m02, m01, m11, m12, m02, m12, m22)
}

/// Compute the Coriolis vector C(q, qdot).
///
/// For the desktop triple-pendulum model:
/// h12 = -(m2+m3)l1l2sin(φ₁)
/// h13 = -m3l1l3sin(φ₁+φ₂)
/// h23 = -m3l2l3sin(φ₂)
/// C[0] = (h12+h13)(2dθ₁+dφ₁)dφ₁ + (h13+h23)(2dθ₁+2dφ₁+dφ₂)dφ₂
/// C[1] = -(h12+h13)dθ₁² + h23(2dθ₁+2dφ₁+dφ₂)dφ₂
/// C[2] = -(h13+h23)dθ₁² - h23(2dθ₁+dφ₁)dφ₁
pub fn coriolis(q: &[f64; 3], qdot: &[f64; 3], params: &TriplePendulumParams) -> SVector<f64, 3> {
    let phi1 = q[1];
    let phi2 = q[2];
    let sum_phi12 = phi1 + phi2;
    let dtheta1 = qdot[0];
    let dphi1 = qdot[1];
    let dphi2 = qdot[2];

    let m23 = params.masses[1] + params.masses[2];
    let m3 = params.masses[2];

    let h12 = -m23 * params.lengths[0] * params.lengths[1] * phi1.sin();
    let h13 = -m3 * params.lengths[0] * params.lengths[2] * sum_phi12.sin();
    let h23 = -m3 * params.lengths[1] * params.lengths[2] * phi2.sin();

    let c0 = (h12 + h13) * (2.0 * dtheta1 + dphi1) * dphi1
        + (h13 + h23) * (2.0 * dtheta1 + 2.0 * dphi1 + dphi2) * dphi2;

    let c1 = -(h12 + h13) * dtheta1 * dtheta1 + h23 * (2.0 * dtheta1 + 2.0 * dphi1 + dphi2) * dphi2;

    let c2 = -(h13 + h23) * dtheta1 * dtheta1 - h23 * (2.0 * dtheta1 + dphi1) * dphi1;

    SVector::<f64, 3>::new(c0, c1, c2)
}

/// Compute the gravity vector G(q).
///
/// For the triple pendulum:
/// θ₂ = θ₁ + φ₁
/// θ₃ = θ₁ + φ₁ + φ₂
///
/// G[0] = (m1+m2+m3)*g*l1*sin(θ₁) + (m2+m3)*g*l2*sin(θ₂) + m3*g*l3*sin(θ₃)
/// G[1] = (m2+m3)*g*l2*sin(θ₂) + m3*g*l3*sin(θ₃)
/// G[2] = m3*g*l3*sin(θ₃)
pub fn gravity_vector(q: &[f64; 3], params: &TriplePendulumParams) -> SVector<f64, 3> {
    let theta1 = q[0];
    let theta2 = theta1 + q[1];
    let theta3 = theta1 + q[1] + q[2];

    let sin_theta1 = theta1.sin();
    let sin_theta2 = theta2.sin();
    let sin_theta3 = theta3.sin();

    let m123 = params.masses[0] + params.masses[1] + params.masses[2];
    let m23 = params.masses[1] + params.masses[2];
    let m3 = params.masses[2];

    let g0 = m123 * params.g * params.lengths[0] * sin_theta1
        + m23 * params.g * params.lengths[1] * sin_theta2
        + m3 * params.g * params.lengths[2] * sin_theta3;

    let g1 = m23 * params.g * params.lengths[1] * sin_theta2
        + m3 * params.g * params.lengths[2] * sin_theta3;

    let g2 = m3 * params.g * params.lengths[2] * sin_theta3;

    SVector::<f64, 3>::new(g0, g1, g2)
}

/// Compute forward kinematics.
///
/// Given configuration q = [θ₁, φ₁, φ₂], compute the positions of:
/// - joint1: end of segment 1
/// - joint2: end of segment 2
/// - joint3: end of segment 3 (tip)
pub fn forward_kinematics(q: &[f64; 3], params: &TriplePendulumParams) -> TripleFKResult {
    let theta1 = q[0];
    let theta2 = theta1 + q[1];
    let theta3 = theta1 + q[1] + q[2];

    // Joint 1: shoulder + l1*(sin(θ₁), -cos(θ₁))
    let joint1 = (
        params.lengths[0] * theta1.sin(),
        -params.lengths[0] * theta1.cos(),
    );

    // Joint 2: joint1 + l2*(sin(θ₂), -cos(θ₂))
    let joint2 = (
        joint1.0 + params.lengths[1] * theta2.sin(),
        joint1.1 - params.lengths[1] * theta2.cos(),
    );

    // Joint 3: joint2 + l3*(sin(θ₃), -cos(θ₃))
    let joint3 = (
        joint2.0 + params.lengths[2] * theta3.sin(),
        joint2.1 - params.lengths[2] * theta3.cos(),
    );

    TripleFKResult {
        joint1,
        joint2,
        joint3,
        angles: [theta1, theta2, theta3],
    }
}

/// Compute the Jacobian of joint positions with respect to q.
///
/// Returns a 2x3 matrix for each joint.
pub fn jacobian_joint1(q: &[f64; 3], params: &TriplePendulumParams) -> SMatrix<f64, 2, 3> {
    let theta1 = q[0];
    let cos_theta1 = theta1.cos();
    let sin_theta1 = theta1.sin();

    let l1 = params.lengths[0];

    SMatrix::<f64, 2, 3>::new(l1 * cos_theta1, 0.0, 0.0, -l1 * sin_theta1, 0.0, 0.0)
}

pub fn jacobian_joint2(q: &[f64; 3], params: &TriplePendulumParams) -> SMatrix<f64, 2, 3> {
    let theta1 = q[0];
    let theta2 = theta1 + q[1];

    let cos_theta1 = theta1.cos();
    let sin_theta1 = theta1.sin();
    let cos_theta2 = theta2.cos();
    let sin_theta2 = theta2.sin();

    let l1 = params.lengths[0];
    let l2 = params.lengths[1];

    let j00 = l1 * cos_theta1 + l2 * cos_theta2;
    let j01 = l2 * cos_theta2;
    let j10 = l1 * sin_theta1 + l2 * sin_theta2;
    let j11 = l2 * sin_theta2;

    SMatrix::<f64, 2, 3>::new(j00, j01, 0.0, j10, j11, 0.0)
}

pub fn jacobian_joint3(q: &[f64; 3], params: &TriplePendulumParams) -> SMatrix<f64, 2, 3> {
    let theta1 = q[0];
    let theta2 = theta1 + q[1];
    let theta3 = theta1 + q[1] + q[2];

    let cos_theta1 = theta1.cos();
    let sin_theta1 = theta1.sin();
    let cos_theta2 = theta2.cos();
    let sin_theta2 = theta2.sin();
    let cos_theta3 = theta3.cos();
    let sin_theta3 = theta3.sin();

    let l1 = params.lengths[0];
    let l2 = params.lengths[1];
    let l3 = params.lengths[2];

    let j00 = l1 * cos_theta1 + l2 * cos_theta2 + l3 * cos_theta3;
    let j01 = l2 * cos_theta2 + l3 * cos_theta3;
    let j02 = l3 * cos_theta3;

    let j10 = l1 * sin_theta1 + l2 * sin_theta2 + l3 * sin_theta3;
    let j11 = l2 * sin_theta2 + l3 * sin_theta3;
    let j12 = l3 * sin_theta3;

    SMatrix::<f64, 2, 3>::new(j00, j01, j02, j10, j11, j12)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_matrix_symmetry() {
        let params = TriplePendulumParams {
            masses: [1.0, 1.0, 1.0],
            lengths: [1.0, 1.0, 1.0],
            g: 9.81,
            friction: [0.0, 0.0, 0.0],
        };

        let q = [0.0, 0.0, 0.0];
        let m = mass_matrix(&q, &params);

        // Mass matrix must be symmetric
        assert!((m.m12 - m.m21).abs() < 1e-12);
        assert!((m.m13 - m.m31).abs() < 1e-12);
        assert!((m.m23 - m.m32).abs() < 1e-12);
    }

    #[test]
    fn test_forward_kinematics_vertical() {
        let params = TriplePendulumParams {
            masses: [1.0, 1.0, 1.0],
            lengths: [1.0, 1.0, 1.0],
            g: 9.81,
            friction: [0.0, 0.0, 0.0],
        };

        let q = [0.0, 0.0, 0.0]; // All hanging straight down
        let fk = forward_kinematics(&q, &params);

        // All joints should be vertically aligned
        assert!(fk.joint1.0.abs() < 1e-12);
        assert!((fk.joint1.1 + 1.0).abs() < 1e-12);
        assert!(fk.joint2.0.abs() < 1e-12);
        assert!((fk.joint2.1 + 2.0).abs() < 1e-12);
        assert!(fk.joint3.0.abs() < 1e-12);
        assert!((fk.joint3.1 + 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mass_matrix_matches_python_aligned_known_value() {
        let params = TriplePendulumParams {
            masses: [1.0, 1.0, 1.0],
            lengths: [1.0, 1.0, 1.0],
            g: 9.81,
            friction: [0.0, 0.0, 0.0],
        };

        let q = [0.0, 0.0, 0.0];
        let m = mass_matrix(&q, &params);

        assert!((m[(0, 0)] - 14.0).abs() < 1e-12);
        assert!((m[(0, 1)] - 8.0).abs() < 1e-12);
        assert!((m[(0, 2)] - 3.0).abs() < 1e-12);
        assert!((m[(1, 1)] - 5.0).abs() < 1e-12);
        assert!((m[(1, 2)] - 2.0).abs() < 1e-12);
        assert!((m[(2, 2)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mass_matrix_matches_python_perpendicular_known_value() {
        let params = TriplePendulumParams {
            masses: [1.0, 1.0, 1.0],
            lengths: [1.0, 1.0, 1.0],
            g: 9.81,
            friction: [0.0, 0.0, 0.0],
        };

        let q = [0.0, std::f64::consts::FRAC_PI_2, 0.0];
        let m = mass_matrix(&q, &params);

        assert!((m[(0, 0)] - 8.0).abs() < 1e-12);
        assert!((m[(0, 1)] - 5.0).abs() < 1e-12);
        assert!((m[(0, 2)] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_coriolis_matches_python_cross_coupling() {
        let params = TriplePendulumParams {
            masses: [1.0, 1.0, 1.0],
            lengths: [1.0, 1.0, 1.0],
            g: 9.81,
            friction: [0.0, 0.0, 0.0],
        };

        let q = [0.0, std::f64::consts::FRAC_PI_2, 0.0];
        let qdot = [1.0, 2.0, 0.0];
        let c = coriolis(&q, &qdot, &params);

        assert!((c[0] + 24.0).abs() < 1e-12);
        assert!((c[1] - 3.0).abs() < 1e-12);
        assert!((c[2] - 1.0).abs() < 1e-12);
    }
}
