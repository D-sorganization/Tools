//! Constraint solver for the golfer model (KKT system, Baumgarte stabilization).

use crate::golfer;
use crate::types::GolferParams;
use nalgebra::{SMatrix, SVector};

const CONSTRAINT_REGULARIZATION: f64 = 1e-9;
const MASS_REGULARIZATION: f64 = 1e-9;

/// Baumgarte stabilization parameters.
#[derive(Debug, Clone, Copy)]
pub struct BaumgarteGains {
    /// Stabilization gain α (controls velocity error correction)
    pub alpha: f64,
    /// Stabilization gain β (controls position error correction)
    pub beta: f64,
}

impl Default for BaumgarteGains {
    fn default() -> Self {
        BaumgarteGains {
            alpha: 10.0,
            beta: 10.0,
        }
    }
}

/// Compute constraint acceleration bias term γ = -dJ/dt * qdot - ∂²Φ/∂q².
///
/// Using finite differences:
/// γ ≈ -(J(q + h*qdot) - J(q - h*qdot)) / (2h) * qdot
pub fn constraint_acceleration_bias(
    q: &[f64; 8],
    qdot: &[f64; 8],
    params: &GolferParams,
) -> SVector<f64, 4> {
    let eps = 1e-7;

    let mut q_plus = *q;
    let mut q_minus = *q;

    for i in 0..8 {
        q_plus[i] += eps * qdot[i];
        q_minus[i] -= eps * qdot[i];
    }

    let j_plus = golfer::constraint_jacobian(&q_plus, params);
    let j_minus = golfer::constraint_jacobian(&q_minus, params);

    let dj_dt = (j_plus - j_minus) / (2.0 * eps);

    let qdot_vec = SVector::<f64, 8>::from_row_slice(qdot);

    -dj_dt * qdot_vec
}

/// Solve the constrained dynamics system using KKT conditions.
///
/// System:
/// M(q) * a + C(q, qdot) + G(q) = τ + J^T * λ
/// Φ(q) = 0  (constraints)
/// J(q) * a + γ(q, qdot) + α * J(q) * qdot + β * Φ(q) = 0  (Baumgarte stabilization)
///
/// Returns the generalized accelerations (a) and Lagrange multipliers (λ).
pub fn constrained_accelerations(
    q: &[f64; 8],
    qdot: &[f64; 8],
    tau: &[f64; 8],
    params: &GolferParams,
    gains: &BaumgarteGains,
) -> (SVector<f64, 8>, SVector<f64, 4>) {
    // Compute mass matrix and dynamics terms
    let m = golfer::mass_matrix(q, params);
    let c = golfer::coriolis(q, qdot, params);
    let g = golfer::gravity_vector(q, params);

    // Constraint terms
    let phi = golfer::constraint_vector(q, params);
    let j = golfer::constraint_jacobian(q, params);
    let gamma = constraint_acceleration_bias(q, qdot, params);

    let tau_vec = SVector::<f64, 8>::from_row_slice(tau);
    let qdot_vec = SVector::<f64, 8>::from_row_slice(qdot);

    // Right-hand side of KKT system:
    // rhs = [τ - C - G; -γ - α*J*qdot - β*Φ]
    let rhs_upper = tau_vec - c - g;
    let rhs_lower = -gamma - gains.alpha * j * qdot_vec - gains.beta * phi;

    // Solve via the Schur complement J M^-1 J^T, which is numerically
    // better behaved than factorizing the full indefinite KKT system.
    let mut regularized_mass = m;
    for i in 0..8 {
        regularized_mass[(i, i)] += MASS_REGULARIZATION;
    }
    let m_inv = regularized_mass
        .try_inverse()
        .expect("Mass matrix singular");
    let mut schur = j * m_inv * j.transpose();
    for i in 0..4 {
        schur[(i, i)] += CONSTRAINT_REGULARIZATION;
    }
    let schur_rhs = rhs_lower - j * m_inv * rhs_upper;
    let lambda = schur
        .lu()
        .solve(&schur_rhs)
        .expect("Constraint system singular");
    let a = m_inv * (rhs_upper + j.transpose() * lambda);

    (a, lambda)
}

/// Project generalized coordinates to the constraint manifold using Newton's method.
///
/// Solves: find Δq such that Φ(q + Δq) ≈ 0 via Newton iteration:
/// Δq_{n+1} = Δq_n - J^{-1} * Φ(q + Δq_n)
pub fn project_to_constraints(
    q: &[f64; 8],
    params: &GolferParams,
    max_iters: usize,
    tol: f64,
) -> [f64; 8] {
    let mut q_proj = *q;

    for _iter in 0..max_iters {
        let phi = golfer::constraint_vector(&q_proj, params);
        let j = golfer::constraint_jacobian(&q_proj, params);

        // Solve J * Δq = -Φ
        let delta_q = solve_minimum_norm_correction(&j, &(-phi)).expect("Jacobian singular");

        // Update
        for i in 0..8 {
            q_proj[i] += delta_q[i];
        }

        // Check convergence
        if phi.norm() < tol {
            break;
        }
    }

    q_proj
}

/// Project velocity to the constraint surface using minimum-norm correction.
///
/// Solves: find Δqdot such that J * Δqdot = -dJ/dt * qdot via least-squares.
pub fn project_velocity(q: &[f64; 8], qdot: &[f64; 8], params: &GolferParams) -> [f64; 8] {
    let j = golfer::constraint_jacobian(q, params);
    let eps = 1e-7;

    let mut q_plus = *q;
    let mut q_minus = *q;

    for i in 0..8 {
        q_plus[i] += eps * qdot[i];
        q_minus[i] -= eps * qdot[i];
    }

    let j_plus = golfer::constraint_jacobian(&q_plus, params);
    let j_minus = golfer::constraint_jacobian(&q_minus, params);

    let dj_dt = (j_plus - j_minus) / (2.0 * eps);
    let qdot_vec = SVector::<f64, 8>::from_row_slice(qdot);

    // Solve: J * Δqdot = -dJ/dt * qdot
    let rhs = -dj_dt * qdot_vec;
    let delta_qdot = solve_minimum_norm_correction(&j, &rhs).expect("Jacobian singular");

    let mut qdot_proj = *qdot;
    for i in 0..8 {
        qdot_proj[i] += delta_qdot[i];
    }

    qdot_proj
}

fn solve_minimum_norm_correction(
    jacobian: &SMatrix<f64, 4, 8>,
    rhs: &SVector<f64, 4>,
) -> Option<SVector<f64, 8>> {
    let normal_matrix = jacobian * jacobian.transpose();
    let lambda = normal_matrix.lu().solve(rhs)?;
    Some(jacobian.transpose() * lambda)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_acceleration_bias_finite_diff() {
        let params = GolferParams {
            l_hub: 0.5,
            m_hub: 5.0,
            d_rs: 0.15,
            d_ls: 0.15,
            l_r_upper: 0.3,
            m_r_upper: 2.0,
            l_r_fore: 0.25,
            m_r_fore: 1.0,
            l_l_upper: 0.3,
            m_l_upper: 2.0,
            l_l_fore: 0.25,
            m_l_fore: 1.0,
            l_club: 1.0,
            m_club: 0.2,
            m_clubhead: 0.2,
            grip_right: 0.3,
            grip_left: 0.3,
            g: 9.81,
            friction: [0.0; 7],
        };

        let q = [0.0; 8];
        let qdot = [0.1; 8];

        let gamma = constraint_acceleration_bias(&q, &qdot, &params);

        // Should be finite and not NaN
        for i in 0..4 {
            assert!(gamma[i].is_finite());
        }
    }

    #[test]
    fn test_constrained_accelerations_kkt() {
        let params = GolferParams {
            l_hub: 0.5,
            m_hub: 5.0,
            d_rs: 0.15,
            d_ls: 0.15,
            l_r_upper: 0.3,
            m_r_upper: 2.0,
            l_r_fore: 0.25,
            m_r_fore: 1.0,
            l_l_upper: 0.3,
            m_l_upper: 2.0,
            l_l_fore: 0.25,
            m_l_fore: 1.0,
            l_club: 1.0,
            m_club: 0.2,
            m_clubhead: 0.2,
            grip_right: 0.3,
            grip_left: 0.3,
            g: 9.81,
            friction: [0.0; 7],
        };

        let q = [0.0; 8];
        let qdot = [0.0; 8];
        let tau = [0.0; 8];
        let gains = BaumgarteGains::default();

        let (a, _lambda) = constrained_accelerations(&q, &qdot, &tau, &params, &gains);

        // Accelerations should be finite
        for i in 0..8 {
            assert!(a[i].is_finite());
        }
    }
}
