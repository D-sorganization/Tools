//! Constraint solver for the golfer model (KKT system, Baumgarte stabilization).

use crate::golfer;
use crate::types::GolferParams;
use nalgebra::{SMatrix, SVector};

/// Baumgarte stabilization parameters.
#[derive(Debug, Clone, Copy)]
pub struct BaumgarteGains {
    /// Stabilization gain α (controls velocity error correction)
    pub alpha: f64,
    /// Stabilization gain β (controls position error correction)
    pub beta: f64,
}

impl BaumgarteGains {
    /// Default Baumgarte gains for typical mechanical systems
    pub fn default() -> Self {
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

    let j_q = golfer::constraint_jacobian(q, params);

    let mut q_plus = *q;
    let mut q_minus = *q;

    for i in 0..8 {
        q_plus[i] += eps * qdot[i];
        q_minus[i] -= eps * qdot[i];
    }

    let j_plus = golfer::constraint_jacobian(&q_plus, params);
    let j_minus = golfer::constraint_jacobian(&q_minus, params);

    let dj_dt = (j_plus - j_minus) / (2.0 * eps);

    let qdot_vec = SVector::from(qdot.as_slice().try_into().unwrap());

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

    let tau_vec = SVector::from(tau.as_slice().try_into().unwrap());
    let qdot_vec = SVector::from(qdot.as_slice().try_into().unwrap());

    // Right-hand side of KKT system:
    // rhs = [τ - C - G; -γ - α*J*qdot - β*Φ]
    let rhs_upper = tau_vec - c - g;
    let rhs_lower = -gamma - gains.alpha * j.clone() * qdot_vec - gains.beta * phi;

    // Build the 12x12 KKT system:
    // [M   J^T] [a] = [τ - C - G]
    // [J    0 ] [λ]   [-γ - α*J*qdot - β*Φ]

    let mut kkt = SMatrix::<f64, 12, 12>::zeros();
    let mut rhs = SVector::<f64, 12>::zeros();

    // M block
    for i in 0..8 {
        for j in 0..8 {
            kkt[(i, j)] = m[(i, j)];
        }
    }

    // J^T block
    for i in 0..8 {
        for j in 0..4 {
            kkt[(i, 8 + j)] = j[(j, i)];
        }
    }

    // J block
    for i in 0..4 {
        for j in 0..8 {
            kkt[(8 + i, j)] = j[(i, j)];
        }
    }

    // RHS
    for i in 0..8 {
        rhs[i] = rhs_upper[i];
    }
    for i in 0..4 {
        rhs[8 + i] = rhs_lower[i];
    }

    // Solve the KKT system via LU decomposition
    let kkt_lu = kkt.lu();
    let sol = kkt_lu.solve(&rhs).expect("KKT system singular");

    let mut a = SVector::<f64, 8>::zeros();
    let mut lambda = SVector::<f64, 4>::zeros();

    for i in 0..8 {
        a[i] = sol[i];
    }
    for i in 0..4 {
        lambda[i] = sol[8 + i];
    }

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
        let j_lu = j.lu();
        let delta_q = -j_lu.solve(&phi).expect("Jacobian singular");

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
pub fn project_velocity(
    q: &[f64; 8],
    qdot: &[f64; 8],
    params: &GolferParams,
) -> [f64; 8] {
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
    let qdot_vec = SVector::from(qdot.as_slice().try_into().unwrap());

    // Solve: J * Δqdot = -dJ/dt * qdot
    let rhs = -dj_dt * qdot_vec;
    let j_lu = j.lu();
    let delta_qdot = j_lu.solve(&rhs).expect("Jacobian singular");

    let mut qdot_proj = *qdot;
    for i in 0..8 {
        qdot_proj[i] += delta_qdot[i];
    }

    qdot_proj
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
