//! Golfer body model (8-DOF with 4 constraints) physics implementation.
//!
//! Generalized coordinates: q = [θ_hub, α_rs, α_re, α_rh, α_ls, α_le, α_lh, θ_club]
//! where:
//! - θ_hub: hub (torso) rotation angle from vertical
//! - α_rs, α_re, α_rh: right arm joint angles (shoulder, elbow, wrist)
//! - α_ls, α_le, α_lh: left arm joint angles (shoulder, elbow, wrist)
//! - θ_club: club angle from vertical
//!
//! Absolute angles:
//! - Right arm: θ_rs = θ_hub + α_rs, θ_re = θ_hub + α_rs + α_re, θ_rh = θ_hub + α_rs + α_re + α_rh
//! - Left arm: θ_ls = θ_hub + α_ls, θ_le = θ_hub + α_ls + α_le, θ_lh = θ_hub + α_ls + α_le + α_lh

use crate::types::{GolferFKResult, GolferParams, Vec2};
use nalgebra::{SMatrix, SVector};
use std::collections::HashMap;

/// Compute forward kinematics and get all joint positions.
///
/// Returns a GolferFKResult containing all 7 mass point positions.
pub fn forward_kinematics(q: &[f64; 8], params: &GolferParams) -> GolferFKResult {
    let theta_hub = q[0];
    let theta_club = q[7];

    // Hub center
    let hub_sin = theta_hub.sin();
    let hub_cos = theta_hub.cos();
    let hub = Vec2::new(params.l_hub * hub_sin, -params.l_hub * hub_cos);

    // Right shoulder
    let rs = hub.add(Vec2::new(
        params.d_rs * hub_cos,
        params.d_rs * hub_sin,
    ));

    // Left shoulder
    let ls = hub.add(Vec2::new(
        -params.d_ls * hub_cos,
        -params.d_ls * hub_sin,
    ));

    // Right arm absolute angles
    let theta_rs = theta_hub + q[1];
    let theta_re = theta_hub + q[1] + q[2];
    let theta_rh = theta_hub + q[1] + q[2] + q[3];

    // Right arm kinematics
    let re = rs.add(Vec2::from_polar(params.l_r_upper, theta_rs));
    let rh = re.add(Vec2::from_polar(params.l_r_fore, theta_re));

    // Left arm absolute angles
    let theta_ls = theta_hub + q[4];
    let theta_le = theta_hub + q[4] + q[5];
    let theta_lh = theta_hub + q[4] + q[5] + q[6];

    // Left arm kinematics
    let le = ls.add(Vec2::from_polar(params.l_l_upper, theta_ls));
    let lh = le.add(Vec2::from_polar(params.l_l_fore, theta_le));

    // Club kinematics
    // Club base is at right wrist minus grip offset along club shaft
    let club_sin = theta_club.sin();
    let club_cos = theta_club.cos();
    let club_base = rh.add(Vec2::new(
        -params.grip_right * club_sin,
        params.grip_right * club_cos,
    ));

    let club_com = club_base.add(Vec2::new(
        (params.l_club / 2.0) * club_sin,
        -(params.l_club / 2.0) * club_cos,
    ));

    let club_tip = club_base.add(Vec2::new(
        params.l_club * club_sin,
        -params.l_club * club_cos,
    ));

    GolferFKResult {
        hub: (hub.x, hub.y),
        r_shoulder: (rs.x, rs.y),
        r_elbow: (re.x, re.y),
        r_wrist: (rh.x, rh.y),
        l_shoulder: (ls.x, ls.y),
        l_elbow: (le.x, le.y),
        l_wrist: (lh.x, lh.y),
        club_base: (club_base.x, club_base.y),
        club_com: (club_com.x, club_com.y),
        club_tip: (club_tip.x, club_tip.y),
    }
}

/// Compute analytical forward kinematics Jacobians for all mass points.
///
/// Returns a HashMap with keys: "hub", "r_shoulder", "r_elbow", "r_wrist",
/// "l_shoulder", "l_elbow", "l_wrist", "club_com", "club_tip"
/// Each value is a 2x8 Jacobian matrix.
pub fn analytical_fk_jacobians(q: &[f64; 8], params: &GolferParams) -> HashMap<String, SMatrix<f64, 2, 8>> {
    let mut jacobians = HashMap::new();

    let theta_hub = q[0];
    let theta_rs = theta_hub + q[1];
    let theta_re = theta_hub + q[1] + q[2];
    let theta_rh = theta_hub + q[1] + q[2] + q[3];
    let theta_ls = theta_hub + q[4];
    let theta_le = theta_hub + q[4] + q[5];
    let theta_lh = theta_hub + q[4] + q[5] + q[6];
    let theta_club = q[7];

    let hub_sin = theta_hub.sin();
    let hub_cos = theta_hub.cos();

    // Hub Jacobian
    let mut j_hub = SMatrix::<f64, 2, 8>::zeros();
    j_hub[(0, 0)] = params.l_hub * hub_cos;
    j_hub[(1, 0)] = params.l_hub * hub_sin;
    jacobians.insert("hub".to_string(), j_hub);

    // Right shoulder Jacobian
    let mut j_rs = SMatrix::<f64, 2, 8>::zeros();
    j_rs[(0, 0)] = params.l_hub * hub_cos - params.d_rs * hub_sin;
    j_rs[(1, 0)] = params.l_hub * hub_sin + params.d_rs * hub_cos;
    jacobians.insert("r_shoulder".to_string(), j_rs);

    // Left shoulder Jacobian
    let mut j_ls = SMatrix::<f64, 2, 8>::zeros();
    j_ls[(0, 0)] = params.l_hub * hub_cos + params.d_ls * hub_sin;
    j_ls[(1, 0)] = params.l_hub * hub_sin - params.d_ls * hub_cos;
    jacobians.insert("l_shoulder".to_string(), j_ls);

    // Right elbow Jacobian
    let mut j_re = SMatrix::<f64, 2, 8>::zeros();
    let cos_rs = theta_rs.cos();
    let sin_rs = theta_rs.sin();
    j_re[(0, 0)] = params.l_hub * hub_cos - params.d_rs * hub_sin + params.l_r_upper * cos_rs;
    j_re[(1, 0)] = params.l_hub * hub_sin + params.d_rs * hub_cos + params.l_r_upper * sin_rs;
    j_re[(0, 1)] = params.l_r_upper * cos_rs;
    j_re[(1, 1)] = params.l_r_upper * sin_rs;
    jacobians.insert("r_elbow".to_string(), j_re);

    // Right wrist Jacobian
    let mut j_rh = SMatrix::<f64, 2, 8>::zeros();
    let cos_re = theta_re.cos();
    let sin_re = theta_re.sin();
    j_rh[(0, 0)] = params.l_hub * hub_cos - params.d_rs * hub_sin
        + params.l_r_upper * cos_rs
        + params.l_r_fore * cos_re;
    j_rh[(1, 0)] = params.l_hub * hub_sin + params.d_rs * hub_cos
        + params.l_r_upper * sin_rs
        + params.l_r_fore * sin_re;
    j_rh[(0, 1)] = params.l_r_upper * cos_rs + params.l_r_fore * cos_re;
    j_rh[(1, 1)] = params.l_r_upper * sin_rs + params.l_r_fore * sin_re;
    j_rh[(0, 2)] = params.l_r_fore * cos_re;
    j_rh[(1, 2)] = params.l_r_fore * sin_re;
    jacobians.insert("r_wrist".to_string(), j_rh);

    // Left elbow Jacobian
    let mut j_le = SMatrix::<f64, 2, 8>::zeros();
    let cos_ls = theta_ls.cos();
    let sin_ls = theta_ls.sin();
    j_le[(0, 0)] = params.l_hub * hub_cos + params.d_ls * hub_sin + params.l_l_upper * cos_ls;
    j_le[(1, 0)] = params.l_hub * hub_sin - params.d_ls * hub_cos + params.l_l_upper * sin_ls;
    j_le[(0, 4)] = params.l_l_upper * cos_ls;
    j_le[(1, 4)] = params.l_l_upper * sin_ls;
    jacobians.insert("l_elbow".to_string(), j_le);

    // Left wrist Jacobian
    let mut j_lh = SMatrix::<f64, 2, 8>::zeros();
    let cos_le = theta_le.cos();
    let sin_le = theta_le.sin();
    j_lh[(0, 0)] = params.l_hub * hub_cos + params.d_ls * hub_sin
        + params.l_l_upper * cos_ls
        + params.l_l_fore * cos_le;
    j_lh[(1, 0)] = params.l_hub * hub_sin - params.d_ls * hub_cos
        + params.l_l_upper * sin_ls
        + params.l_l_fore * sin_le;
    j_lh[(0, 4)] = params.l_l_upper * cos_ls + params.l_l_fore * cos_le;
    j_lh[(1, 4)] = params.l_l_upper * sin_ls + params.l_l_fore * sin_le;
    j_lh[(0, 5)] = params.l_l_fore * cos_le;
    j_lh[(1, 5)] = params.l_l_fore * sin_le;
    jacobians.insert("l_wrist".to_string(), j_lh);

    // Club COM Jacobian
    let mut j_club_com = SMatrix::<f64, 2, 8>::zeros();
    let club_sin = theta_club.sin();
    let club_cos = theta_club.cos();
    let cos_rh = theta_rh.cos();
    let sin_rh = theta_rh.sin();
    let half_club = params.l_club / 2.0;

    j_club_com[(0, 0)] = params.l_hub * hub_cos - params.d_rs * hub_sin
        + params.l_r_upper * cos_rs
        + params.l_r_fore * cos_re
        - params.grip_right * club_sin
        + half_club * club_sin;
    j_club_com[(1, 0)] = params.l_hub * hub_sin + params.d_rs * hub_cos
        + params.l_r_upper * sin_rs
        + params.l_r_fore * sin_re
        + params.grip_right * club_cos
        - half_club * club_cos;
    j_club_com[(0, 1)] = params.l_r_upper * cos_rs + params.l_r_fore * cos_re;
    j_club_com[(1, 1)] = params.l_r_upper * sin_rs + params.l_r_fore * sin_re;
    j_club_com[(0, 2)] = params.l_r_fore * cos_re;
    j_club_com[(1, 2)] = params.l_r_fore * sin_re;
    j_club_com[(0, 3)] = 0.0;
    j_club_com[(1, 3)] = 0.0;
    j_club_com[(0, 7)] = -params.grip_right * club_cos + half_club * club_cos;
    j_club_com[(1, 7)] = -params.grip_right * (-club_sin) - half_club * (-club_sin);
    jacobians.insert("club_com".to_string(), j_club_com);

    // Club tip Jacobian
    let mut j_club_tip = SMatrix::<f64, 2, 8>::zeros();
    j_club_tip[(0, 0)] = params.l_hub * hub_cos - params.d_rs * hub_sin
        + params.l_r_upper * cos_rs
        + params.l_r_fore * cos_re
        - params.grip_right * club_sin
        + params.l_club * club_sin;
    j_club_tip[(1, 0)] = params.l_hub * hub_sin + params.d_rs * hub_cos
        + params.l_r_upper * sin_rs
        + params.l_r_fore * sin_re
        + params.grip_right * club_cos
        - params.l_club * club_cos;
    j_club_tip[(0, 1)] = params.l_r_upper * cos_rs + params.l_r_fore * cos_re;
    j_club_tip[(1, 1)] = params.l_r_upper * sin_rs + params.l_r_fore * sin_re;
    j_club_tip[(0, 2)] = params.l_r_fore * cos_re;
    j_club_tip[(1, 2)] = params.l_r_fore * sin_re;
    j_club_tip[(0, 3)] = 0.0;
    j_club_tip[(1, 3)] = 0.0;
    j_club_tip[(0, 7)] = -params.grip_right * club_cos + params.l_club * club_cos;
    j_club_tip[(1, 7)] = -params.grip_right * (-club_sin) - params.l_club * (-club_sin);
    jacobians.insert("club_tip".to_string(), j_club_tip);

    jacobians
}

/// Compute the 8x8 mass matrix M(q) via M = Σ m_i * J_i^T * J_i.
pub fn mass_matrix(q: &[f64; 8], params: &GolferParams) -> SMatrix<f64, 8, 8> {
    let jacobians = analytical_fk_jacobians(q, params);

    let mut m = SMatrix::<f64, 8, 8>::zeros();

    // Hub
    if let Some(j) = jacobians.get("hub") {
        m += params.m_hub * j.transpose() * j;
    }

    // Right shoulder (shoulder point is part of torso structure, included in hub)
    // Right elbow (intermediate)
    if let Some(j) = jacobians.get("r_elbow") {
        m += params.m_r_upper * j.transpose() * j;
    }

    // Right wrist
    if let Some(j) = jacobians.get("r_wrist") {
        m += params.m_r_fore * j.transpose() * j;
    }

    // Left elbow (intermediate)
    if let Some(j) = jacobians.get("l_elbow") {
        m += params.m_l_upper * j.transpose() * j;
    }

    // Left wrist
    if let Some(j) = jacobians.get("l_wrist") {
        m += params.m_l_fore * j.transpose() * j;
    }

    // Club COM
    if let Some(j) = jacobians.get("club_com") {
        m += params.m_club * j.transpose() * j;
    }

    // Club head
    if let Some(j) = jacobians.get("club_tip") {
        m += params.m_clubhead * j.transpose() * j;
    }

    m
}

/// Compute the Coriolis vector C(q, qdot) = -dM/dt * qdot + M * d²/dt²(J) * qdot.
///
/// For now, we use a finite-difference approximation:
/// C ≈ (M(q + eps*qdot) - M(q - eps*qdot)) / (2*eps) * qdot
///
/// This is slower but numerically robust. A more efficient analytical form would
/// require computing all time derivatives of the Jacobians.
pub fn coriolis(q: &[f64; 8], qdot: &[f64; 8], params: &GolferParams) -> SVector<f64, 8> {
    let eps = 1e-7;

    let mut q_plus = [0.0; 8];
    let mut q_minus = [0.0; 8];

    for i in 0..8 {
        q_plus[i] = q[i] + eps * qdot[i];
        q_minus[i] = q[i] - eps * qdot[i];
    }

    let m_plus = mass_matrix(&q_plus, params);
    let m_minus = mass_matrix(&q_minus, params);

    let dm_dqdot = (m_plus - m_minus) / (2.0 * eps);

    -dm_dqdot * SVector::from(qdot.as_slice().try_into().unwrap())
}

/// Compute the gravity vector G(q).
///
/// G_i = -Σ_j (m_j * g) * (∂y_j / ∂q_i)
/// where y_j is the vertical (y-axis) position of mass j.
pub fn gravity_vector(q: &[f64; 8], params: &GolferParams) -> SVector<f64, 8> {
    let jacobians = analytical_fk_jacobians(q, params);
    let mut g = SVector::<f64, 8>::zeros();

    // For each mass point, extract the vertical component of its Jacobian
    // and add -m_j * g * J_y to the gravity vector

    if let Some(j) = jacobians.get("hub") {
        for i in 0..8 {
            g[i] -= params.m_hub * params.g * j[(1, i)];
        }
    }

    if let Some(j) = jacobians.get("r_elbow") {
        for i in 0..8 {
            g[i] -= params.m_r_upper * params.g * j[(1, i)];
        }
    }

    if let Some(j) = jacobians.get("r_wrist") {
        for i in 0..8 {
            g[i] -= params.m_r_fore * params.g * j[(1, i)];
        }
    }

    if let Some(j) = jacobians.get("l_elbow") {
        for i in 0..8 {
            g[i] -= params.m_l_upper * params.g * j[(1, i)];
        }
    }

    if let Some(j) = jacobians.get("l_wrist") {
        for i in 0..8 {
            g[i] -= params.m_l_fore * params.g * j[(1, i)];
        }
    }

    if let Some(j) = jacobians.get("club_com") {
        for i in 0..8 {
            g[i] -= params.m_club * params.g * j[(1, i)];
        }
    }

    if let Some(j) = jacobians.get("club_tip") {
        for i in 0..8 {
            g[i] -= params.m_clubhead * params.g * j[(1, i)];
        }
    }

    g
}

/// Compute the 4-dimensional constraint vector Φ(q).
///
/// Constraints:
/// 1. Right wrist constrained to left wrist (hands grip club together)
/// 2. Left hand position on club shaft
pub fn constraint_vector(q: &[f64; 8], params: &GolferParams) -> SVector<f64, 4> {
    let fk = forward_kinematics(q, params);

    let rh_x = fk.r_wrist.0;
    let rh_y = fk.r_wrist.1;
    let lh_x = fk.l_wrist.0;
    let lh_y = fk.l_wrist.1;

    let club_base_x = fk.club_base.0;
    let club_base_y = fk.club_base.1;
    let club_tip_x = fk.club_tip.0;
    let club_tip_y = fk.club_tip.1;

    // Constraint 1 & 2: Right and left hands at the same location
    let c1 = rh_x - lh_x;
    let c2 = rh_y - lh_y;

    // Constraint 3 & 4: Left hand on club shaft (parameterized by grip_left offset)
    // Left hand should be at: club_base + grip_left * (unit vector along shaft)
    let club_dx = club_tip_x - club_base_x;
    let club_dy = club_tip_y - club_base_y;
    let club_len_sq = club_dx * club_dx + club_dy * club_dy;
    let club_len = club_len_sq.sqrt();

    let expected_lh_x = club_base_x + (params.grip_left / club_len) * club_dx;
    let expected_lh_y = club_base_y + (params.grip_left / club_len) * club_dy;

    let c3 = lh_x - expected_lh_x;
    let c4 = lh_y - expected_lh_y;

    SVector::<f64, 4>::new(c1, c2, c3, c4)
}

/// Compute the 4x8 constraint Jacobian ∂Φ/∂q.
pub fn constraint_jacobian(q: &[f64; 8], params: &GolferParams) -> SMatrix<f64, 4, 8> {
    let eps = 1e-7;

    let phi_q = constraint_vector(q, params);

    let mut jac = SMatrix::<f64, 4, 8>::zeros();

    for j in 0..8 {
        let mut q_pert = *q;
        q_pert[j] += eps;
        let phi_pert = constraint_vector(&q_pert, params);

        for i in 0..4 {
            jac[(i, j)] = (phi_pert[i] - phi_q[i]) / eps;
        }
    }

    jac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_kinematics_identity() {
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

        let q = [0.0; 8]; // All angles zero
        let fk = forward_kinematics(&q, &params);

        // Hub should be at (0, -l_hub)
        assert!((fk.hub.0).abs() < 1e-10);
        assert!((fk.hub.1 + params.l_hub).abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_dimensions() {
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
        let jacobians = analytical_fk_jacobians(&q, &params);

        assert_eq!(jacobians.get("hub").unwrap().nrows(), 2);
        assert_eq!(jacobians.get("hub").unwrap().ncols(), 8);
    }

    #[test]
    fn test_mass_matrix_symmetry() {
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
        let m = mass_matrix(&q, &params);

        // Mass matrix should be symmetric
        for i in 0..8 {
            for j in 0..8 {
                assert!((m[(i, j)] - m[(j, i)]).abs() < 1e-10);
            }
        }
    }
}
