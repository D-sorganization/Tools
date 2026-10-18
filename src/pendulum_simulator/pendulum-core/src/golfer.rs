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

    // Hub center.  The standoff extends *upward* from the origin (inside the
    // arm loop), parity with the Python model (`golfer_kinematics._hub_position`,
    // #1103): hub = (-L sin(theta), +L cos(theta)).
    let hub_sin = theta_hub.sin();
    let hub_cos = theta_hub.cos();
    let hub = Vec2::new(-params.l_hub * hub_sin, params.l_hub * hub_cos);

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
    j_hub[(0, 0)] = -params.l_hub * hub_cos;
    j_hub[(1, 0)] = -params.l_hub * hub_sin;
    jacobians.insert("hub".to_string(), j_hub);

    // Right shoulder Jacobian
    let mut j_rs = SMatrix::<f64, 2, 8>::zeros();
    j_rs[(0, 0)] = -params.l_hub * hub_cos - params.d_rs * hub_sin;
    j_rs[(1, 0)] = -params.l_hub * hub_sin + params.d_rs * hub_cos;
    jacobians.insert("r_shoulder".to_string(), j_rs);

    // Left shoulder Jacobian
    let mut j_ls = SMatrix::<f64, 2, 8>::zeros();
    j_ls[(0, 0)] = -params.l_hub * hub_cos + params.d_ls * hub_sin;
    j_ls[(1, 0)] = -params.l_hub * hub_sin - params.d_ls * hub_cos;
    jacobians.insert("l_shoulder".to_string(), j_ls);

    // Right elbow Jacobian
    let mut j_re = SMatrix::<f64, 2, 8>::zeros();
    let cos_rs = theta_rs.cos();
    let sin_rs = theta_rs.sin();
    j_re[(0, 0)] = -params.l_hub * hub_cos - params.d_rs * hub_sin + params.l_r_upper * cos_rs;
    j_re[(1, 0)] = -params.l_hub * hub_sin + params.d_rs * hub_cos + params.l_r_upper * sin_rs;
    j_re[(0, 1)] = params.l_r_upper * cos_rs;
    j_re[(1, 1)] = params.l_r_upper * sin_rs;
    jacobians.insert("r_elbow".to_string(), j_re);

    // Right wrist Jacobian
    let mut j_rh = SMatrix::<f64, 2, 8>::zeros();
    let cos_re = theta_re.cos();
    let sin_re = theta_re.sin();
    j_rh[(0, 0)] = -params.l_hub * hub_cos - params.d_rs * hub_sin
        + params.l_r_upper * cos_rs
        + params.l_r_fore * cos_re;
    j_rh[(1, 0)] = -params.l_hub * hub_sin
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
    j_le[(0, 0)] = -params.l_hub * hub_cos + params.d_ls * hub_sin + params.l_l_upper * cos_ls;
    j_le[(1, 0)] = -params.l_hub * hub_sin - params.d_ls * hub_cos + params.l_l_upper * sin_ls;
    j_le[(0, 4)] = params.l_l_upper * cos_ls;
    j_le[(1, 4)] = params.l_l_upper * sin_ls;
    jacobians.insert("l_elbow".to_string(), j_le);

    // Left wrist Jacobian
    let mut j_lh = SMatrix::<f64, 2, 8>::zeros();
    let cos_le = theta_le.cos();
    let sin_le = theta_le.sin();
    j_lh[(0, 0)] = -params.l_hub * hub_cos
        + params.d_ls * hub_sin
        + params.l_l_upper * cos_ls
        + params.l_l_fore * cos_le;
    j_lh[(1, 0)] = -params.l_hub * hub_sin - params.d_ls * hub_cos
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

/// Central-difference step for `dM/dq` in [`coriolis`].  Near the double-precision
/// optimum (`eps^(1/3) ~ 6e-6`).  Keep equal to `CORIOLIS_FD_STEP` in
/// `golfer_dynamics.py`.
pub const CORIOLIS_FD_STEP: f64 = 1e-5;

/// Contract the Christoffel symbols of the first kind with `qdot` twice.
///
/// `dm[k]` is `dM/dq_k`.  With `c_ijk = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)`
/// the Coriolis/centrifugal vector is
///
/// ```text
/// C_i = sum_jk c_ijk * qdot_j * qdot_k
///     = (Mdot qdot)_i - 0.5 * qdot^T (dM/dq_i) qdot
/// ```
///
/// because the first two terms of `c_ijk` each contribute `0.5 * (Mdot qdot)_i`
/// (they are equal after relabelling `j <-> k`) and the third is the gradient of
/// the quadratic form.  This is the single definition shared with the Python
/// `golfer_dynamics.christoffel_first_kind` (#5640); the Python contraction
/// `0.5 * (dM + dM.transpose(0, 2, 1) - dM.transpose(2, 0, 1))` is the same tensor.
///
/// Precondition: every `dm[k]` is symmetric (derivative of a symmetric `M`).
pub(crate) fn christoffel_contraction(
    dm: &[SMatrix<f64, 8, 8>; 8],
    qdot: &SVector<f64, 8>,
) -> SVector<f64, 8> {
    let mut m_dot_qdot = SVector::<f64, 8>::zeros();
    let mut kinetic_gradient = SVector::<f64, 8>::zeros();
    for (k, dm_dqk) in dm.iter().enumerate() {
        let dm_qdot = dm_dqk * qdot;
        m_dot_qdot += qdot[k] * dm_qdot;
        kinetic_gradient[k] = qdot.dot(&dm_qdot);
    }
    m_dot_qdot - 0.5 * kinetic_gradient
}

/// Compute the Coriolis/centrifugal vector C(q, qdot).
///
/// Parity with the Python model (`golfer_dynamics.analytical_coriolis`, #5638,
/// #5640):
///
/// ```text
/// C_i = sum_jk 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) * qdot_j * qdot_k
/// ```
///
/// with `dM/dq_k` a central difference (step [`CORIOLIS_FD_STEP`]) of the
/// analytical mass matrix.  See [`christoffel_contraction`] for the closed form
/// actually evaluated.  Until #5640 both backends dropped the third term (they
/// used `dM_ki/dq_j`, which equals the second term for a symmetric `M`) and
/// returned `0.5 * Mdot * qdot`; the two were fixed together to stay one solver.
/// The step moved from a `1e-7` forward difference to a `1e-5` central one in the
/// same change: round-off in `M` divided by `1e-7` made the backends disagree at
/// ~1e-8 relative, the central difference brings that to ~1e-11.
pub fn coriolis(q: &[f64; 8], qdot: &[f64; 8], params: &GolferParams) -> SVector<f64, 8> {
    let qdot_vec = SVector::<f64, 8>::from_row_slice(qdot);

    let mut dm = [SMatrix::<f64, 8, 8>::zeros(); 8];
    for (k, dm_dqk) in dm.iter_mut().enumerate() {
        let mut q_plus = *q;
        let mut q_minus = *q;
        q_plus[k] += CORIOLIS_FD_STEP;
        q_minus[k] -= CORIOLIS_FD_STEP;
        *dm_dqk = (mass_matrix(&q_plus, params) - mass_matrix(&q_minus, params))
            / (2.0 * CORIOLIS_FD_STEP);
    }
    christoffel_contraction(&dm, &qdot_vec)
}

/// Compute the gravity vector G(q) = dV/dq.
///
/// ```text
/// V = sum_j m_j * g * y_j        =>   G_i = sum_j m_j * g * dy_j/dq_i
/// ```
///
/// where `y_j` is the vertical (y-axis) position of mass `j`.  The equations of
/// motion are `M a + C + G = tau + Phi_q^T lambda`, so `G` enters the
/// right-hand side as `-G`.
pub fn gravity_vector(q: &[f64; 8], params: &GolferParams) -> SVector<f64, 8> {
    let jacobians = analytical_fk_jacobians(q, params);
    let masses = [
        ("hub", params.m_hub),
        ("r_elbow", params.m_r_upper),
        ("r_wrist", params.m_r_fore),
        ("l_elbow", params.m_l_upper),
        ("l_wrist", params.m_l_fore),
        ("club_com", params.m_club),
        ("club_tip", params.m_clubhead),
    ];

    let mut g = SVector::<f64, 8>::zeros();
    for (name, mass) in masses {
        if let Some(j) = jacobians.get(name) {
            for i in 0..8 {
                g[i] += mass * params.g * j[(1, i)];
            }
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

/// Rows of [`constraint_vector`] that form an independent closure (#5635,
/// #5638): the left-wrist pin residual in the world frame.
///
/// Rows 2-3 are the same residual expressed in the club frame, so they add no
/// rank on the constraint manifold and off it differ from rows 0-1 only by a
/// term proportional to `Phi`.  The KKT solver builds its system from these
/// rows only.  This is the single Rust definition; the Python solver keeps its
/// own copy (`golfer_constraints.INDEPENDENT_CLOSURE_ROWS`) and the parity
/// tests assert the two are equal (exposed to Python as
/// `golfer_independent_closure_rows`).
pub const INDEPENDENT_CLOSURE_ROWS: [usize; 2] = [0, 1];

/// Number of closure equations in [`constraint_vector`].
pub const N_CONSTRAINTS: usize = 4;

/// Evaluate the 4 loop-closure equations (parity with Python
/// `golfer_constraints.constraint_vector`, #5638).
///
/// With `d = (sin th_club, -cos th_club)` the club direction,
/// `n = (cos th_club, sin th_club)` its perpendicular and
/// `s = grip_left - grip_right` the signed grip separation:
///
/// ```text
/// Phi[0:2] = LH - (RH + s d)          (world frame)
/// Phi[2]   = (LH - RH) . n            (perpendicular offset)
/// Phi[3]   = (LH - RH) . d - s        (along-club offset)
/// ```
///
/// The club is anchored at the right wrist, so rows 2-3 restate rows 0-1 in
/// the club frame: only [`INDEPENDENT_CLOSURE_ROWS`] are independent.
pub fn constraint_vector(q: &[f64; 8], params: &GolferParams) -> SVector<f64, 4> {
    let fk = forward_kinematics(q, params);
    let (sin_club, cos_club) = q[7].sin_cos();
    let club_dir = Vec2::new(sin_club, -cos_club);
    let club_perp = Vec2::new(cos_club, sin_club);
    let grip_sep = params.grip_left - params.grip_right;

    let rh_to_lh = Vec2::new(fk.l_wrist.0 - fk.r_wrist.0, fk.l_wrist.1 - fk.r_wrist.1);

    SVector::<f64, 4>::new(
        rh_to_lh.x - grip_sep * club_dir.x,
        rh_to_lh.y - grip_sep * club_dir.y,
        rh_to_lh.x * club_perp.x + rh_to_lh.y * club_perp.y,
        rh_to_lh.x * club_dir.x + rh_to_lh.y * club_dir.y - grip_sep,
    )
}

/// Analytical 4x8 closure Jacobian `dPhi/dq` (parity with Python
/// `golfer_constraints.analytical_constraint_jacobian`, #5638).
pub fn constraint_jacobian(q: &[f64; 8], params: &GolferParams) -> SMatrix<f64, 4, 8> {
    let jacobians = analytical_fk_jacobians(q, params);
    let j_rel = jacobians["l_wrist"] - jacobians["r_wrist"]; // d(LH - RH)/dq, 2x8

    let fk = forward_kinematics(q, params);
    let (sin_club, cos_club) = q[7].sin_cos();
    let club_dir = SVector::<f64, 2>::new(sin_club, -cos_club);
    let club_perp = SVector::<f64, 2>::new(cos_club, sin_club);
    let d_club_dir = SVector::<f64, 2>::new(cos_club, sin_club); // d(club_dir)/d(th_club)
    let d_club_perp = SVector::<f64, 2>::new(-sin_club, cos_club);
    let grip_sep = params.grip_left - params.grip_right;
    let rh_to_lh = SVector::<f64, 2>::new(fk.l_wrist.0 - fk.r_wrist.0, fk.l_wrist.1 - fk.r_wrist.1);

    let mut jac = SMatrix::<f64, 4, 8>::zeros();
    for col in 0..8 {
        jac[(0, col)] = j_rel[(0, col)];
        jac[(1, col)] = j_rel[(1, col)];
        jac[(2, col)] = club_perp.dot(&j_rel.column(col));
        jac[(3, col)] = club_dir.dot(&j_rel.column(col));
    }
    jac[(0, 7)] -= grip_sep * d_club_dir[0];
    jac[(1, 7)] -= grip_sep * d_club_dir[1];
    jac[(2, 7)] += rh_to_lh.dot(&d_club_perp);
    jac[(3, 7)] += rh_to_lh.dot(&d_club_dir);
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

        // Hub extends upward from the origin (Python parity, #1103/#5638)
        assert!((fk.hub.0).abs() < 1e-10);
        assert!((fk.hub.1 - params.l_hub).abs() < 1e-10);
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
        // The closure Jacobian is analytical (#5638, parity with Python
        // `analytical_constraint_jacobian`), so it must match a central
        // difference of Phi to the same FD_TOL as the FK Jacobians.
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
                            (jac[(row, col)] - d[row]).abs() < FD_TOL,
                            "Phi[{row}] d/dq{col}: {} vs {}",
                            jac[(row, col)],
                            d[row]
                        );
                    }
                }
            }
        }
    }

    fn potential_energy(q: &[f64; 8], params: &GolferParams) -> f64 {
        let fk = forward_kinematics(q, params);
        params.g
            * (params.m_hub * fk.hub.1
                + params.m_r_upper * fk.r_elbow.1
                + params.m_r_fore * fk.r_wrist.1
                + params.m_l_upper * fk.l_elbow.1
                + params.m_l_fore * fk.l_wrist.1
                + params.m_club * fk.club_com.1
                + params.m_clubhead * fk.club_tip.1)
    }

    #[test]
    fn test_gravity_vector_is_the_potential_gradient() {
        // G = dV/dq (the EOM is M a + C + G = tau + Phi_q^T lambda).  Before
        // #5638 the sign was inverted, so a Rust-backed solve was anti-gravity.
        let params = params_with_grip_right(0.3);
        for q in NON_TRIVIAL_CONFIGS {
            let g = gravity_vector(&q, &params);
            for col in 0..8 {
                let mut qp = q;
                let mut qm = q;
                qp[col] += FD_STEP;
                qm[col] -= FD_STEP;
                let d = (potential_energy(&qp, &params) - potential_energy(&qm, &params))
                    / (2.0 * FD_STEP);
                assert!(
                    (g[col] - d).abs() < 1e-6,
                    "G[{col}] = {} vs dV/dq = {d}",
                    g[col]
                );
            }
        }
    }

    #[test]
    fn test_constraint_vector_vanishes_on_the_closed_loop() {
        // Place the left wrist exactly at RH + (grip_left - grip_right) * dir:
        // all four closure rows must vanish (rows 2-3 restate rows 0-1).
        let params = params_with_grip_right(0.05);
        let q = NON_TRIVIAL_CONFIGS[0];
        let phi = constraint_vector(&q, &params);
        let fk = forward_kinematics(&q, &params);
        let sep = params.grip_left - params.grip_right;
        let (s, c) = q[7].sin_cos();
        let expected_row0 = fk.l_wrist.0 - (fk.r_wrist.0 + sep * s);
        let expected_row1 = fk.l_wrist.1 - (fk.r_wrist.1 - sep * c);
        assert!((phi[0] - expected_row0).abs() < 1e-12);
        assert!((phi[1] - expected_row1).abs() < 1e-12);
        assert_eq!(INDEPENDENT_CLOSURE_ROWS, [0, 1]);
    }

    // ---- Coriolis / Christoffel (#5640) -------------------------------------

    const QDOT_SAMPLES: [[f64; 8]; 3] = [
        [0.6, -1.1, 0.4, 0.0, 0.9, -0.7, 0.0, 1.3],
        [-1.4, 0.3, 1.0, 0.0, -0.5, 0.8, 0.0, -0.9],
        [0.2, 0.7, -0.6, 0.0, 1.2, 1.1, 0.0, 0.5],
    ];

    /// Deterministic dM[k] stack, each symmetric (derivative of a symmetric M).
    fn symmetric_derivative_stack() -> [SMatrix<f64, 8, 8>; 8] {
        let mut dm = [SMatrix::<f64, 8, 8>::zeros(); 8];
        for (k, m) in dm.iter_mut().enumerate() {
            for i in 0..8 {
                for j in i..8 {
                    let v = ((1 + i) as f64 * 0.37 + (1 + j) as f64 * 0.91 + (1 + k) as f64 * 1.7)
                        .sin();
                    m[(i, j)] = v;
                    m[(j, i)] = v;
                }
            }
        }
        dm
    }

    #[test]
    fn test_christoffel_contraction_matches_brute_force_definition() {
        let dm = symmetric_derivative_stack();
        for qd in QDOT_SAMPLES {
            let qdot = SVector::<f64, 8>::from_row_slice(&qd);
            let fast = christoffel_contraction(&dm, &qdot);
            for i in 0..8 {
                let mut expected = 0.0;
                for j in 0..8 {
                    for k in 0..8 {
                        // c_ijk = 0.5 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)
                        let c = 0.5 * (dm[k][(i, j)] + dm[j][(i, k)] - dm[i][(j, k)]);
                        expected += c * qd[j] * qd[k];
                    }
                }
                assert!(
                    (fast[i] - expected).abs() < 1e-12,
                    "C[{i}] = {} vs brute force {expected}",
                    fast[i]
                );
            }
        }
    }

    /// Euler-Lagrange reference built from M(q) alone (no Christoffel algebra):
    /// `C = d/dt (M qdot) - dT/dq` with `T = 0.5 qdot^T M qdot`, central differences.
    fn euler_lagrange_coriolis(
        q: &[f64; 8],
        qdot: &[f64; 8],
        params: &GolferParams,
    ) -> SVector<f64, 8> {
        let qd = SVector::<f64, 8>::from_row_slice(qdot);
        let mut qp = *q;
        let mut qm = *q;
        for i in 0..8 {
            qp[i] += FD_STEP * qdot[i];
            qm[i] -= FD_STEP * qdot[i];
        }
        let momentum_rate =
            (mass_matrix(&qp, params) * qd - mass_matrix(&qm, params) * qd) / (2.0 * FD_STEP);
        let kinetic = |qq: &[f64; 8]| 0.5 * qd.dot(&(mass_matrix(qq, params) * qd));
        let mut grad_t = SVector::<f64, 8>::zeros();
        for i in 0..8 {
            let mut a = *q;
            let mut b = *q;
            a[i] += FD_STEP;
            b[i] -= FD_STEP;
            grad_t[i] = (kinetic(&a) - kinetic(&b)) / (2.0 * FD_STEP);
        }
        momentum_rate - grad_t
    }

    #[test]
    fn test_coriolis_matches_euler_lagrange_reference() {
        // Before #5640 this returned 0.5 * Mdot qdot (relative error 0.5-1.1).
        let params = params_with_grip_right(0.3);
        for q in NON_TRIVIAL_CONFIGS {
            for qd in QDOT_SAMPLES {
                let c = coriolis(&q, &qd, &params);
                let reference = euler_lagrange_coriolis(&q, &qd, &params);
                let err = (c - reference).norm() / reference.norm();
                assert!(err < 1e-5, "relative error {err} for q={q:?} qdot={qd:?}");
            }
        }
    }

    #[test]
    fn test_coriolis_is_not_half_mdot_qdot() {
        // Guard the exact #5640 failure mode.
        let params = params_with_grip_right(0.3);
        let q = NON_TRIVIAL_CONFIGS[0];
        let qd = QDOT_SAMPLES[0];
        let qdot = SVector::<f64, 8>::from_row_slice(&qd);
        let mut m_dot_qdot = SVector::<f64, 8>::zeros();
        for k in 0..8 {
            let mut qp = q;
            let mut qm = q;
            qp[k] += FD_STEP;
            qm[k] -= FD_STEP;
            let dm = (mass_matrix(&qp, &params) - mass_matrix(&qm, &params)) / (2.0 * FD_STEP);
            m_dot_qdot += qd[k] * (dm * qdot);
        }
        let c = coriolis(&q, &qd, &params);
        assert!((c - 0.5 * m_dot_qdot).norm() > 1e-3 * c.norm());
    }

    #[test]
    fn test_coriolis_power_balance() {
        // qdot^T (Mdot - 2C) qdot = 0 (skew-symmetry of Mdot - 2C(q, qdot)).
        let params = params_with_grip_right(0.3);
        for q in NON_TRIVIAL_CONFIGS {
            for qd in QDOT_SAMPLES {
                let qdot = SVector::<f64, 8>::from_row_slice(&qd);
                let mut m_dot = SMatrix::<f64, 8, 8>::zeros();
                for k in 0..8 {
                    let mut qp = q;
                    let mut qm = q;
                    qp[k] += FD_STEP;
                    qm[k] -= FD_STEP;
                    m_dot += qd[k] * (mass_matrix(&qp, &params) - mass_matrix(&qm, &params))
                        / (2.0 * FD_STEP);
                }
                let c = coriolis(&q, &qd, &params);
                let balance = qdot.dot(&(m_dot * qdot - 2.0 * c));
                assert!(balance.abs() < 1e-5, "power balance residual {balance}");
            }
        }
    }
}
