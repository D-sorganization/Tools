//! Triple pendulum (3-DOF) physics implementation.
//!
//! Model:
//! - q = [θ₁, φ₂, φ₃] where θ₁ is shoulder angle, φ₂ is elbow relative, φ₃ is wrist relative
//! - Three segments with masses m₁, m₂, m₃ and lengths l₁, l₂, l₃

use crate::types::{TripleFKResult, TriplePendulumParams};
use nalgebra::{SMatrix, SVector};

/// Compute the 3x3 mass matrix M(q).
///
/// For the triple pendulum, the mass matrix is:
/// M[0,0] = (m1 + m2 + m3) * l1²
/// M[0,1] = (m2 + m3) * l1 * l2 * cos(φ₂)
/// M[0,2] = m3 * l1 * l3 * cos(φ₂ + φ₃)
/// M[1,1] = (m2 + m3) * l2²
/// M[1,2] = m3 * l2 * l3 * cos(φ₃)
/// M[2,2] = m3 * l3²
/// (symmetric)
pub fn mass_matrix(q: &[f64; 3], params: &TriplePendulumParams) -> SMatrix<f64, 3, 3> {
    let phi2 = q[1];
    let phi3 = q[2];
    let sum_phi23 = phi2 + phi3;

    let cos_phi2 = phi2.cos();
    let cos_phi3 = phi3.cos();
    let cos_sum_phi23 = sum_phi23.cos();

    let l1_sq = params.lengths[0] * params.lengths[0];
    let l2_sq = params.lengths[1] * params.lengths[1];
    let l3_sq = params.lengths[2] * params.lengths[2];

    let m00 = (params.masses[0] + params.masses[1] + params.masses[2]) * l1_sq;
    let m01 = (params.masses[1] + params.masses[2])
        * params.lengths[0]
        * params.lengths[1]
        * cos_phi2;
    let m02 = params.masses[2] * params.lengths[0] * params.lengths[2] * cos_sum_phi23;
    let m11 = (params.masses[1] + params.masses[2]) * l2_sq;
    let m12 = params.masses[2] * params.lengths[1] * params.lengths[2] * cos_phi3;
    let m22 = params.masses[2] * l3_sq;

    SMatrix::<f64, 3, 3>::new(m00, m01, m02, m01, m11, m12, m02, m12, m22)
}

/// Compute the Coriolis vector C(q, qdot).
///
/// For the triple pendulum:
/// C[0] = -(m2+m3)*l1*l2*sin(φ₂)*qdot[1]² - m3*l1*l3*sin(φ₂+φ₃)*qdot[2]²
///        - 2*m3*l1*l3*sin(φ₂+φ₃)*qdot[1]*qdot[2]
/// C[1] = (m2+m3)*l1*l2*sin(φ₂)*qdot[0]² - m3*l2*l3*sin(φ₃)*qdot[2]²
///        - 2*m3*l2*l3*sin(φ₃)*qdot[1]*qdot[2]
/// C[2] = m3*l1*l3*sin(φ₂+φ₃)*qdot[0]² + m3*l2*l3*sin(φ₃)*qdot[1]²
pub fn coriolis(q: &[f64; 3], qdot: &[f64; 3], params: &TriplePendulumParams) -> SVector<f64, 3> {
    let phi2 = q[1];
    let phi3 = q[2];
    let sum_phi23 = phi2 + phi3;

    let sin_phi2 = phi2.sin();
    let sin_phi3 = phi3.sin();
    let sin_sum_phi23 = sum_phi23.sin();

    let m23 = params.masses[1] + params.masses[2];
    let m3 = params.masses[2];

    let term_l1l2_phi2 = params.lengths[0] * params.lengths[1] * sin_phi2;
    let term_l1l3_sum = params.lengths[0] * params.lengths[2] * sin_sum_phi23;
    let term_l2l3_phi3 = params.lengths[1] * params.lengths[2] * sin_phi3;

    let c0 = -m23 * term_l1l2_phi2 * qdot[1] * qdot[1]
        - m3 * term_l1l3_sum * (qdot[2] * qdot[2] + 2.0 * qdot[1] * qdot[2]);

    let c1 = m23 * term_l1l2_phi2 * qdot[0] * qdot[0]
        - m3 * term_l2l3_phi3 * (qdot[2] * qdot[2] + 2.0 * qdot[1] * qdot[2]);

    let c2 = m3 * term_l1l3_sum * qdot[0] * qdot[0] + m3 * term_l2l3_phi3 * qdot[1] * qdot[1];

    SVector::<f64, 3>::new(c0, c1, c2)
}

/// Compute the gravity vector G(q).
///
/// For the triple pendulum:
/// θ₂ = θ₁ + φ₂
/// θ₃ = θ₁ + φ₂ + φ₃
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
/// Given configuration q = [θ₁, φ₂, φ₃], compute the positions of:
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
}
