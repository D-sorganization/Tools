//! Double pendulum (2-DOF) physics implementation.
//!
//! Model:
//! - q = [θ₁, φ] where θ₁ is arm angle from vertical, φ is club angle relative to arm
//! - Absolute club angle: θ₂ = θ₁ + φ
//! - Segment 1: shoulder to wrist (arm), length L1, mass m1
//! - Segment 2: wrist to tip (club), length L2, mass m2

use crate::types::{DoubleFKResult, DoublePendulumParams};
use nalgebra::{SMatrix, SVector};

/// Compute the 2x2 mass matrix M(q).
///
/// For the double pendulum with relative coordinates:
/// M[0,0] = (m1 + m2) * L1²
/// M[0,1] = m2 * L1 * L2 * cos(φ)
/// M[1,0] = M[0,1]
/// M[1,1] = m2 * L2²
pub fn mass_matrix(q: &[f64; 2], params: &DoublePendulumParams) -> SMatrix<f64, 2, 2> {
    let phi = q[1];
    let cos_phi = phi.cos();

    let m00 = (params.m1 + params.m2) * params.l1 * params.l1;
    let m01 = params.m2 * params.l1 * params.l2 * cos_phi;
    let m11 = params.m2 * params.l2 * params.l2;

    SMatrix::<f64, 2, 2>::new(m00, m01, m01, m11)
}

/// Compute the Coriolis vector C(q, qdot).
///
/// For the double pendulum:
/// C[0] = -m2 * L1 * L2 * sin(φ) * qdot[1]²
/// C[1] = m2 * L1 * L2 * sin(φ) * qdot[0]²
pub fn coriolis(q: &[f64; 2], qdot: &[f64; 2], params: &DoublePendulumParams) -> SVector<f64, 2> {
    let phi = q[1];
    let sin_phi = phi.sin();
    let c_term = params.m2 * params.l1 * params.l2 * sin_phi;

    let c0 = -c_term * qdot[1] * qdot[1];
    let c1 = c_term * qdot[0] * qdot[0];

    SVector::<f64, 2>::new(c0, c1)
}

/// Compute the gravity vector G(q).
///
/// For the double pendulum:
/// G[0] = (m1 + m2) * g * L1 * sin(θ₁)
/// G[1] = m2 * g * L2 * sin(θ₁ + φ)
pub fn gravity_vector(q: &[f64; 2], params: &DoublePendulumParams) -> SVector<f64, 2> {
    let theta1 = q[0];
    let phi = q[1];
    let theta2 = theta1 + phi;

    let g0 = (params.m1 + params.m2) * params.g * params.l1 * theta1.sin();
    let g1 = params.m2 * params.g * params.l2 * theta2.sin();

    SVector::<f64, 2>::new(g0, g1)
}

/// Compute forward kinematics.
///
/// Given configuration q = [θ₁, φ], compute the positions of:
/// - Wrist (end of segment 1)
/// - Club tip (end of segment 2)
pub fn forward_kinematics(q: &[f64; 2], params: &DoublePendulumParams) -> DoubleFKResult {
    let theta1 = q[0];
    let phi = q[1];
    let theta2 = theta1 + phi;

    // Wrist position: shoulder + L1*(sin(θ₁), -cos(θ₁))
    let wrist = (
        params.l1 * theta1.sin(),
        -params.l1 * theta1.cos(),
    );

    // Club tip: wrist + L2*(sin(θ₂), -cos(θ₂))
    let club_tip = (
        wrist.0 + params.l2 * theta2.sin(),
        wrist.1 - params.l2 * theta2.cos(),
    );

    DoubleFKResult {
        wrist,
        club_tip,
        theta1,
        theta2,
    }
}

/// Compute the Jacobian of the wrist position with respect to q.
///
/// wrist(q) = [L1*sin(θ₁), -L1*cos(θ₁)]
/// J_wrist = [ [L1*cos(θ₁),     0    ],
///             [-L1*sin(θ₁),     0    ] ]
pub fn jacobian_wrist(q: &[f64; 2], params: &DoublePendulumParams) -> SMatrix<f64, 2, 2> {
    let theta1 = q[0];
    let cos_theta1 = theta1.cos();
    let sin_theta1 = theta1.sin();

    SMatrix::<f64, 2, 2>::new(
        params.l1 * cos_theta1,
        0.0,
        -params.l1 * sin_theta1,
        0.0,
    )
}

/// Compute the Jacobian of the club tip position with respect to q.
///
/// club_tip(q) = [L1*sin(θ₁) + L2*sin(θ₁+φ), -L1*cos(θ₁) - L2*cos(θ₁+φ)]
/// ∂(club_tip)/∂θ₁ = [L1*cos(θ₁) + L2*cos(θ₁+φ), L1*sin(θ₁) + L2*sin(θ₁+φ)]
/// ∂(club_tip)/∂φ = [L2*cos(θ₁+φ), L2*sin(θ₁+φ)]
pub fn jacobian_club_tip(q: &[f64; 2], params: &DoublePendulumParams) -> SMatrix<f64, 2, 2> {
    let theta1 = q[0];
    let phi = q[1];
    let theta2 = theta1 + phi;

    let cos_theta1 = theta1.cos();
    let sin_theta1 = theta1.sin();
    let cos_theta2 = theta2.cos();
    let sin_theta2 = theta2.sin();

    let j00 = params.l1 * cos_theta1 + params.l2 * cos_theta2;
    let j10 = params.l1 * sin_theta1 + params.l2 * sin_theta2;
    let j01 = params.l2 * cos_theta2;
    let j11 = params.l2 * sin_theta2;

    SMatrix::<f64, 2, 2>::new(j00, j01, j10, j11)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_matrix_validity() {
        let params = DoublePendulumParams {
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
            friction1: 0.0,
            friction2: 0.0,
        };

        let q = [0.0, 0.0];
        let m = mass_matrix(&q, &params);

        // At φ=0: M should be symmetric and positive definite
        assert!((m.m12 - m.m21).abs() < 1e-12);
        assert!(m.m11 > 0.0);
        assert!(m.m22 > 0.0);
    }

    #[test]
    fn test_forward_kinematics() {
        let params = DoublePendulumParams {
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
            friction1: 0.0,
            friction2: 0.0,
        };

        let q = [0.0, 0.0]; // Hanging down
        let fk = forward_kinematics(&q, &params);

        // Both segments hanging straight down
        assert!(fk.wrist.0.abs() < 1e-12);
        assert!((fk.wrist.1 + 1.0).abs() < 1e-12);
        assert!(fk.club_tip.0.abs() < 1e-12);
        assert!((fk.club_tip.1 + 2.0).abs() < 1e-12);
    }
}
