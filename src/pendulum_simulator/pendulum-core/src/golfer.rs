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
    let rs = hub.add(Vec2::new(params.d_rs * hub_cos, params.d_rs * hub_sin));

    // Left shoulder
    let ls = hub.add(Vec2::new(-params.d_ls * hub_cos, -params.d_ls * hub_sin));

    // Right arm absolute angles
    let theta_rs = theta_hub + q[1];
    let theta_re = theta_hub + q[1] + q[2];

    // Right arm kinematics
    let re = rs.add(Vec2::from_polar(params.l_r_upper, theta_rs));
    let rh = re.add(Vec2::from_polar(params.l_r_fore, theta_re));

    // Left arm absolute angles
    let theta_ls = theta_hub + q[4];
    let theta_le = theta_hub + q[4] + q[5];

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
pub fn analytical_fk_jacobians(
    q: &[f64; 8],
    params: &GolferParams,
) -> HashMap<String, SMatrix<f64, 2, 8>> {
    let mut jacobians = HashMap::new();

    let theta_hub = q[0];
    let theta_rs = theta_hub + q[1];
    let theta_re = theta_hub + q[1] + q[2];
    let theta_ls = theta_hub + q[4];
    let theta_le = theta_hub + q[4] + q[5];
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
    j_rh[(1, 0)] = params.l_hub * hub_sin
        + params.d_rs * hub_cos
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
    j_lh[(0, 0)] = params.l_hub * hub_cos
        + params.d_ls * hub_sin
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

    // Club points are rh + s * (sin, -cos)(theta_club) with s measured from the
    // right wrist pin (s = L/2 - grip_right for the COM, L - grip_right for the
    // tip).  Hence every column except the club angle equals the right wrist's
    // (the club adds no dependence on the hub or arm angles), and the club-angle
    // column is s * (cos, sin)(theta_club).  See issue #5626.
    let club_tangent = SVector::<f64, 2>::new(theta_club.cos(), theta_club.sin());

    let mut j_club_com = j_rh;
    j_club_com
        .column_mut(7)
        .copy_from(&((params.l_club / 2.0 - params.grip_right) * club_tangent));
    jacobians.insert("club_com".to_string(), j_club_com);

    let mut j_club_tip = j_rh;
    j_club_tip
        .column_mut(7)
        .copy_from(&((params.l_club - params.grip_right) * club_tangent));
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

    -dm_dqdot * SVector::<f64, 8>::from_row_slice(qdot)
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

/// Compute viscous friction torque vector for 7 actuated joints.
///
/// τ_friction[i] = -friction_i * qdot[i]  for i = 0..6 (actuated DOFs)
/// τ_friction[7] = 0.0  (club angle is constrained, not actuated)
pub fn friction_torque(qdot: &[f64; 8], params: &GolferParams) -> SVector<f64, 8> {
    let mut f = SVector::<f64, 8>::zeros();
    for i in 0..7 {
        f[i] = -params.friction[i] * qdot[i];
    }
    // qdot[7] (club angle) has no direct friction — it's a constrained DOF
    f
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
            friction: [0.0; 7],
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
            friction: [0.0; 7],
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
            friction: [0.0; 7],
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

    // ---- Jacobian-vs-forward-kinematics consistency (issue #5626) ----
    //
    // Every Jacobian returned by `analytical_fk_jacobians` must equal the
    // derivative of the matching `forward_kinematics` point.  The reference is
    // a central difference with step FD_STEP.  Error analysis: truncation is
    // (h^2/6)|f'''| <= (1e-12/6) * (largest link length ~1.1 m) ~ 2e-13 and
    // round-off is eps*|f|/h ~ 2.2e-16 * 2 / 1e-6 ~ 4e-10, so FD_TOL = 1e-8
    // leaves a >20x margin while still catching any O(1e-3) analytic error
    // (the #5626 bugs are O(0.1)).

    const FD_STEP: f64 = 1e-6;
    const FD_TOL: f64 = 1e-8;

    fn params_with_grip_right(grip_right: f64) -> GolferParams {
        GolferParams {
            l_hub: 0.5,
            m_hub: 5.0,
            d_rs: 0.15,
            d_ls: 0.2,
            l_r_upper: 0.3,
            m_r_upper: 2.0,
            l_r_fore: 0.25,
            m_r_fore: 1.0,
            l_l_upper: 0.33,
            m_l_upper: 2.0,
            l_l_fore: 0.27,
            m_l_fore: 1.0,
            l_club: 1.1,
            m_club: 0.4,
            m_clubhead: 0.2,
            grip_right,
            grip_left: 0.1,
            g: 9.81,
            friction: [0.0; 7],
        }
    }

    fn fk_point(name: &str, q: &[f64; 8], params: &GolferParams) -> (f64, f64) {
        let fk = forward_kinematics(q, params);
        match name {
            "hub" => fk.hub,
            "r_shoulder" => fk.r_shoulder,
            "r_elbow" => fk.r_elbow,
            "r_wrist" => fk.r_wrist,
            "l_shoulder" => fk.l_shoulder,
            "l_elbow" => fk.l_elbow,
            "l_wrist" => fk.l_wrist,
            "club_com" => fk.club_com,
            "club_tip" => fk.club_tip,
            other => panic!("unknown point {other}"),
        }
    }

    const NON_TRIVIAL_CONFIGS: [[f64; 8]; 4] = [
        [0.3, -0.5, 0.8, 0.2, 0.4, 0.7, -0.3, 1.1],
        [-0.7, 0.9, -0.4, -0.6, -0.2, 1.2, 0.5, -0.8],
        [1.2, 0.1, 0.3, 0.9, -1.0, -0.6, 0.7, 2.3],
        [-0.2, -1.1, 1.0, -0.4, 0.8, 0.3, 0.1, 0.4],
    ];

    const JACOBIAN_POINTS: [&str; 9] = [
        "hub",
        "r_shoulder",
        "r_elbow",
        "r_wrist",
        "l_shoulder",
        "l_elbow",
        "l_wrist",
        "club_com",
        "club_tip",
    ];

    fn assert_jacobians_match_fd(grip_right: f64) {
        let params = params_with_grip_right(grip_right);
        for q in NON_TRIVIAL_CONFIGS {
            let jacobians = analytical_fk_jacobians(&q, &params);
            for name in JACOBIAN_POINTS {
                let jac = jacobians.get(name).unwrap();
                for col in 0..8 {
                    let mut qp = q;
                    let mut qm = q;
                    qp[col] += FD_STEP;
                    qm[col] -= FD_STEP;
                    let (xp, yp) = fk_point(name, &qp, &params);
                    let (xm, ym) = fk_point(name, &qm, &params);
                    let fd_x = (xp - xm) / (2.0 * FD_STEP);
                    let fd_y = (yp - ym) / (2.0 * FD_STEP);
                    assert!(
                        (jac[(0, col)] - fd_x).abs() < FD_TOL,
                        "{name} dx/dq{col} (grip_right={grip_right}, q={q:?}): analytic {} vs FD {fd_x}",
                        jac[(0, col)]
                    );
                    assert!(
                        (jac[(1, col)] - fd_y).abs() < FD_TOL,
                        "{name} dy/dq{col} (grip_right={grip_right}, q={q:?}): analytic {} vs FD {fd_y}",
                        jac[(1, col)]
                    );
                }
            }
        }
    }

    #[test]
    fn test_fk_jacobians_match_central_difference_positive_grip_right() {
        assert_jacobians_match_fd(0.3);
    }

    #[test]
    fn test_fk_jacobians_match_central_difference_negative_grip_right() {
        assert_jacobians_match_fd(-0.2);
    }

    #[test]
    fn test_fk_jacobians_match_central_difference_zero_grip_right() {
        assert_jacobians_match_fd(0.0);
    }

    #[test]
    fn test_club_hub_column_equals_right_wrist_hub_column() {
        // A club point is rh + s*u(theta_club): its hub column must be the
        // right wrist's, with no club terms (issue #5626 (b)).
        let params = params_with_grip_right(0.3);
        let q = NON_TRIVIAL_CONFIGS[0];
        let jacobians = analytical_fk_jacobians(&q, &params);
        let j_rh = jacobians.get("r_wrist").unwrap();
        for name in ["club_com", "club_tip"] {
            let j = jacobians.get(name).unwrap();
            for row in 0..2 {
                assert!(
                    (j[(row, 0)] - j_rh[(row, 0)]).abs() < 1e-12,
                    "{name} row {row}"
                );
            }
        }
    }

    #[test]
    fn test_club_angle_column_matches_independent_reference() {
        // Independent reference from the corrected UpstreamDrift formula
        // (UD #12668 `_club_jacobians`): column 7 = s * (cos, sin)(theta_club)
        // with s = L/2 - grip_right (COM) and L - grip_right (tip).
        let params = params_with_grip_right(0.3);
        let q = NON_TRIVIAL_CONFIGS[0];
        let jacobians = analytical_fk_jacobians(&q, &params);
        let (c, s) = (q[7].cos(), q[7].sin());
        let com = 0.5 * params.l_club - params.grip_right;
        let tip = params.l_club - params.grip_right;
        let j_com = jacobians.get("club_com").unwrap();
        let j_tip = jacobians.get("club_tip").unwrap();
        assert!((j_com[(0, 7)] - com * c).abs() < 1e-12);
        assert!((j_com[(1, 7)] - com * s).abs() < 1e-12);
        assert!((j_tip[(0, 7)] - tip * c).abs() < 1e-12);
        assert!((j_tip[(1, 7)] - tip * s).abs() < 1e-12);
    }

    #[test]
    fn test_constraint_jacobian_matches_central_difference_of_constraint_vector() {
        // The closure Jacobian is a forward difference (eps = 1e-7, O(eps)
        // truncation ~ 1e-7 * |Phi''| ~ 1e-7); compare to a central difference
        // of Phi with a 1e-5 tolerance (>> 1e-7 truncation, << O(0.1) bugs).
        for grip_right in [0.3, -0.2] {
            let params = params_with_grip_right(grip_right);
            for q in NON_TRIVIAL_CONFIGS {
                let jac = constraint_jacobian(&q, &params);
                for col in 0..8 {
                    let mut qp = q;
                    let mut qm = q;
                    qp[col] += FD_STEP;
                    qm[col] -= FD_STEP;
                    let d = (constraint_vector(&qp, &params) - constraint_vector(&qm, &params))
                        / (2.0 * FD_STEP);
                    for row in 0..4 {
                        assert!(
                            (jac[(row, col)] - d[row]).abs() < 1e-5,
                            "Phi[{row}] d/dq{col}: {} vs {}",
                            jac[(row, col)],
                            d[row]
                        );
                    }
                }
            }
        }
    }
}
