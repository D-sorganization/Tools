//! Constraint solver for the golfer model (KKT system, Baumgarte stabilization).
//!
//! This is the Rust twin of the Python solver in
//! `double_pendulum_golf.constraint_solver` and must stay one solver with it
//! (#5635, #5638):
//!
//! * the KKT system is built from [`INDEPENDENT_CLOSURE_ROWS`] only -- rows
//!   2-3 of the closure restate rows 0-1 in the club frame, so the full
//!   4-row closure Jacobian has rank 2 on the manifold;
//! * the multipliers on the redundant rows are exactly zero;
//! * independent rows that lose rank raise [`ClosureError::RankLoss`]
//!   (surfaced to Python as `ValueError`, like the Python solver) instead of
//!   returning meaningless multipliers;
//! * the Baumgarte right-hand side is `-gamma - 2 alpha Phi_dot - beta^2 Phi`.
//!
//! Design by Contract: inputs are validated ([`ClosureError::NonFinite`]), and
//! the solution is a minimum-norm least-squares solve of the (singular,
//! because of the massless wrist DOFs) KKT matrix, exactly like the Python
//! solver.

use crate::golfer;
pub use crate::golfer::INDEPENDENT_CLOSURE_ROWS;
use crate::golfer::N_CONSTRAINTS;
use crate::types::GolferParams;
use nalgebra::{SMatrix, SVector};
use std::fmt;

/// Smallest admissible `sigma_min / sigma_max` of the independent closure
/// rows.  Healthy closed configurations sit at O(0.1-1); below this the wrist
/// pin is at a kinematic singularity and the multipliers are not determined.
/// Mirrors Python `constraint_solver.CLOSURE_RANK_RTOL`.
pub const CLOSURE_RANK_RTOL: f64 = 1e-8;

/// Tikhonov term of the minimum-norm projections (Python
/// `project_to_constraints` / `project_velocity`).  Required because the
/// closure Jacobian is rank 2 of 4 rows, so `J J^T` alone is singular.
const PROJECTION_REGULARIZATION: f64 = 1e-12;

/// Forward-difference step of the constraint acceleration bias (Python
/// `_constraint_acceleration_bias`).
const BIAS_FD_STEP: f64 = 1e-7;

const N_DOF: usize = 8;
const N_INDEPENDENT: usize = INDEPENDENT_CLOSURE_ROWS.len();
const N_KKT: usize = N_DOF + N_INDEPENDENT;

/// Contract violations of the closure solver.  Mapped to Python `ValueError`.
#[derive(Debug, Clone, PartialEq)]
pub enum ClosureError {
    /// An input contained NaN or infinity.
    NonFinite(&'static str),
    /// The independent closure rows lost rank (kinematic singularity).
    RankLoss {
        /// Largest singular value of the independent closure rows.
        sigma_max: f64,
        /// Smallest singular value of the independent closure rows.
        sigma_min: f64,
    },
}

impl fmt::Display for ClosureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClosureError::NonFinite(name) => write!(f, "{name} must be finite"),
            ClosureError::RankLoss {
                sigma_max,
                sigma_min,
            } => write!(
                f,
                "independent closure rows lost rank (kinematic singularity): singular values [{sigma_max:e}, {sigma_min:e}]"
            ),
        }
    }
}

impl std::error::Error for ClosureError {}

/// Baumgarte stabilization parameters.
#[derive(Debug, Clone, Copy)]
pub struct BaumgarteGains {
    /// Stabilization gain alpha (velocity error correction, enters as `2 alpha`)
    pub alpha: f64,
    /// Stabilization gain beta (position error correction, enters as `beta^2`)
    pub beta: f64,
}

impl Default for BaumgarteGains {
    /// Parity with Python `DEFAULT_ALPHA` / `DEFAULT_BETA`.
    fn default() -> Self {
        BaumgarteGains {
            alpha: 5.0,
            beta: 5.0,
        }
    }
}

/// Compute constraint acceleration bias term `gamma = d(Phi_q)/dt * qdot`.
///
/// Forward difference of the analytical closure Jacobian along `qdot`
/// (identical to Python `_constraint_acceleration_bias`):
/// `gamma ~ (Phi_q(q + h qdot) - Phi_q(q)) / h * qdot`.
pub fn constraint_acceleration_bias(
    q: &[f64; 8],
    qdot: &[f64; 8],
    params: &GolferParams,
) -> SVector<f64, 4> {
    let mut q_plus = *q;
    for i in 0..N_DOF {
        q_plus[i] += BIAS_FD_STEP * qdot[i];
    }

    let j_0 = golfer::constraint_jacobian(q, params);
    let j_plus = golfer::constraint_jacobian(&q_plus, params);
    let dj_dt = (j_plus - j_0) / BIAS_FD_STEP;

    dj_dt * SVector::<f64, 8>::from_row_slice(qdot)
}

fn require_finite<const R: usize, const C: usize>(
    name: &'static str,
    value: &SMatrix<f64, R, C>,
) -> Result<(), ClosureError> {
    if value.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(ClosureError::NonFinite(name))
    }
}

/// Solve the KKT system on [`INDEPENDENT_CLOSURE_ROWS`] only.
///
/// ```text
/// [ M          Phi_q[r]^T ] [ qddot    ]   [ rhs_dyn           ]
/// [ Phi_q[r]   0          ] [ lambda_r ] = [ rhs_constraint[r] ]
/// ```
///
/// The massless wrist DOFs leave exact zero columns in `M` and `Phi_q`, so the
/// KKT matrix stays singular even on the independent rows; the minimum-norm
/// least-squares solution is exact for that consistent system (same cutoff as
/// `numpy.linalg.lstsq(rcond=None)`).
///
/// # Errors
///
/// * [`ClosureError::NonFinite`] for NaN/inf inputs;
/// * [`ClosureError::RankLoss`] when the independent rows have
///   `sigma_min <= CLOSURE_RANK_RTOL * sigma_max`.
///
/// # Returns
///
/// Accelerations and all four multipliers, zero on the redundant rows.
pub fn solve_independent_kkt(
    mass: &SMatrix<f64, 8, 8>,
    phi_q: &SMatrix<f64, 4, 8>,
    rhs_dyn: &SVector<f64, 8>,
    rhs_constraint: &SVector<f64, 4>,
) -> Result<(SVector<f64, 8>, SVector<f64, 4>), ClosureError> {
    require_finite("M", mass)?;
    require_finite("Phi_q", phi_q)?;
    require_finite("rhs_dyn", rhs_dyn)?;
    require_finite("rhs_constraint", rhs_constraint)?;

    let mut closure = SMatrix::<f64, N_INDEPENDENT, N_DOF>::zeros();
    let mut rhs_closure = SVector::<f64, N_INDEPENDENT>::zeros();
    for (i, &row) in INDEPENDENT_CLOSURE_ROWS.iter().enumerate() {
        closure.set_row(i, &phi_q.row(row));
        rhs_closure[i] = rhs_constraint[row];
    }

    let singular_values = closure.svd(false, false).singular_values;
    let sigma_max = singular_values.max();
    let sigma_min = singular_values.min();
    // Written as a positive comparison so NaN singular values also fail it.
    let full_rank = sigma_min > CLOSURE_RANK_RTOL * sigma_max;
    if !full_rank {
        return Err(ClosureError::RankLoss {
            sigma_max,
            sigma_min,
        });
    }

    let mut kkt = SMatrix::<f64, N_KKT, N_KKT>::zeros();
    kkt.fixed_view_mut::<N_DOF, N_DOF>(0, 0).copy_from(mass);
    kkt.fixed_view_mut::<N_DOF, N_INDEPENDENT>(0, N_DOF)
        .copy_from(&closure.transpose());
    kkt.fixed_view_mut::<N_INDEPENDENT, N_DOF>(N_DOF, 0)
        .copy_from(&closure);
    let mut rhs = SVector::<f64, N_KKT>::zeros();
    rhs.fixed_rows_mut::<N_DOF>(0).copy_from(rhs_dyn);
    rhs.fixed_rows_mut::<N_INDEPENDENT>(N_DOF)
        .copy_from(&rhs_closure);

    let svd = kkt.svd(true, true);
    // numpy lstsq(rcond=None): cutoff = eps * max(M, N) * sigma_max.
    let cutoff = f64::EPSILON * N_KKT as f64 * svd.singular_values.max();
    let solution = svd
        .solve(&rhs, cutoff)
        .map_err(|_| ClosureError::NonFinite("KKT solution"))?;

    let mut lambda = SVector::<f64, N_CONSTRAINTS>::zeros();
    for (i, &row) in INDEPENDENT_CLOSURE_ROWS.iter().enumerate() {
        lambda[row] = solution[N_DOF + i];
    }
    let qddot = solution.fixed_rows::<N_DOF>(0).into_owned();
    Ok((qddot, lambda))
}

/// Solve the constrained dynamics system using KKT conditions.
///
/// System (`G = dV/dq`):
///
/// ```text
/// M(q) a + C(q, qdot) + G(q) = tau + Phi_q^T lambda
/// Phi_q a = -gamma - 2 alpha Phi_q qdot - beta^2 Phi      (Baumgarte)
/// ```
///
/// restricted to [`INDEPENDENT_CLOSURE_ROWS`] (see [`solve_independent_kkt`]).
/// Viscous joint friction is not modelled here; callers must only route
/// frictionless parameter sets to this solver (Python
/// `golfer_native_constraint_dynamics_supported`).
///
/// Returns the generalized accelerations and the four Lagrange multipliers
/// (zero on the redundant rows).
///
/// # Errors
///
/// See [`solve_independent_kkt`].
pub fn constrained_accelerations(
    q: &[f64; 8],
    qdot: &[f64; 8],
    tau: &[f64; 8],
    params: &GolferParams,
    gains: &BaumgarteGains,
) -> Result<(SVector<f64, 8>, SVector<f64, 4>), ClosureError> {
    let m = golfer::mass_matrix(q, params);
    let c = golfer::coriolis(q, qdot, params);
    let g = golfer::gravity_vector(q, params);

    let phi = golfer::constraint_vector(q, params);
    let j = golfer::constraint_jacobian(q, params);
    let gamma = constraint_acceleration_bias(q, qdot, params);

    let tau_vec = SVector::<f64, 8>::from_row_slice(tau);
    let qdot_vec = SVector::<f64, 8>::from_row_slice(qdot);

    let rhs_dyn = tau_vec - c - g;
    let rhs_constraint =
        -gamma - 2.0 * gains.alpha * (j * qdot_vec) - gains.beta * gains.beta * phi;

    solve_independent_kkt(&m, &j, &rhs_dyn, &rhs_constraint)
}

/// `J^T (J J^T + eps I)^-1 rhs`: the regularized minimum-norm correction.
fn minimum_norm_correction(
    jacobian: &SMatrix<f64, 4, 8>,
    rhs: &SVector<f64, 4>,
) -> SVector<f64, 8> {
    let mut normal = jacobian * jacobian.transpose();
    for i in 0..N_CONSTRAINTS {
        normal[(i, i)] += PROJECTION_REGULARIZATION;
    }
    let lambda = normal
        .lu()
        .solve(rhs)
        .expect("regularized closure normal matrix is non-singular");
    jacobian.transpose() * lambda
}

/// Project generalized coordinates to the constraint manifold using Newton's
/// method (parity with Python `project_to_constraints`):
/// `q <- q - J^T (J J^T + eps I)^-1 Phi(q)` until `||Phi|| < tol`.
pub fn project_to_constraints(
    q: &[f64; 8],
    params: &GolferParams,
    max_iters: usize,
    tol: f64,
) -> [f64; 8] {
    let mut q_proj = *q;

    for _iter in 0..max_iters {
        let phi = golfer::constraint_vector(&q_proj, params);
        if phi.norm() < tol {
            break;
        }
        let j = golfer::constraint_jacobian(&q_proj, params);
        let delta_q = minimum_norm_correction(&j, &phi);
        for i in 0..N_DOF {
            q_proj[i] -= delta_q[i];
        }
    }

    q_proj
}

/// Project velocity onto the constraint surface `Phi_q qdot = 0` with the
/// minimum-norm correction `qdot - J^T (J J^T + eps I)^-1 J qdot` (parity with
/// Python `project_velocity`).
pub fn project_velocity(q: &[f64; 8], qdot: &[f64; 8], params: &GolferParams) -> [f64; 8] {
    let j = golfer::constraint_jacobian(q, params);
    let qdot_vec = SVector::<f64, 8>::from_row_slice(qdot);
    let correction = minimum_norm_correction(&j, &(j * qdot_vec));

    let mut qdot_proj = *qdot;
    for i in 0..N_DOF {
        qdot_proj[i] -= correction[i];
    }
    qdot_proj
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> GolferParams {
        GolferParams {
            l_hub: 0.15,
            m_hub: 2.0,
            d_rs: 0.20,
            d_ls: 0.20,
            l_r_upper: 0.35,
            m_r_upper: 3.0,
            l_r_fore: 0.30,
            m_r_fore: 2.0,
            l_l_upper: 0.35,
            m_l_upper: 3.0,
            l_l_fore: 0.30,
            m_l_fore: 2.0,
            l_club: 1.1,
            m_club: 0.5,
            m_clubhead: 0.2,
            grip_right: 0.05,
            grip_left: 0.25,
            g: 9.81,
            friction: [0.0; 7],
        }
    }

    /// Deterministic pseudo-random configuration in [-1, 1] (no rand dependency).
    fn config(seed: u64) -> [f64; 8] {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut out = [0.0; 8];
        for v in out.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        }
        out
    }

    fn closed_state(seed: u64, params: &GolferParams) -> ([f64; 8], [f64; 8]) {
        let q = project_to_constraints(&config(seed), params, 50, 1e-10);
        let mut qdot = config(seed + 1000);
        for v in qdot.iter_mut() {
            *v *= 2.0;
        }
        (q, project_velocity(&q, &qdot, params))
    }

    #[test]
    fn independent_rows_are_the_world_frame_wrist_pin() {
        assert_eq!(INDEPENDENT_CLOSURE_ROWS, [0, 1]);
    }

    #[test]
    fn test_constraint_acceleration_bias_finite_diff() {
        let params = params();
        let q = [0.0; 8];
        let qdot = [0.1; 8];

        let gamma = constraint_acceleration_bias(&q, &qdot, &params);

        for i in 0..4 {
            assert!(gamma[i].is_finite());
        }
    }

    #[test]
    fn closure_jacobian_has_rank_two_and_independent_rows_are_full_rank() {
        let params = params();
        for seed in 0..8 {
            let (q, _) = closed_state(seed, &params);
            let j = golfer::constraint_jacobian(&q, &params);
            let sv = j.svd(false, false).singular_values;
            let mut sorted: Vec<f64> = sv.iter().copied().collect();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            assert!(sorted[2] < 1e-6 * sorted[0], "rank > 2: {sorted:?}");
            assert!(sorted[1] > 1e-3 * sorted[0], "rank < 2: {sorted:?}");
        }
    }

    #[test]
    fn redundant_rows_carry_exactly_zero_multiplier() {
        let params = params();
        let tau = [0.1, 0.2, -0.1, 0.0, 0.3, 0.1, 0.0, 0.0];
        for seed in 0..8 {
            let (q, qdot) = closed_state(seed, &params);
            let (a, lambda) =
                constrained_accelerations(&q, &qdot, &tau, &params, &BaumgarteGains::default())
                    .expect("closed configuration is solvable");
            assert!(a.iter().all(|v| v.is_finite()));
            assert_eq!(lambda[2], 0.0);
            assert_eq!(lambda[3], 0.0);
        }
    }

    #[test]
    fn independent_rows_close_at_acceleration_level() {
        let params = params();
        let tau = [0.1, 0.2, -0.1, 0.0, 0.3, 0.1, 0.0, 0.0];
        let gains = BaumgarteGains::default();
        for seed in 0..8 {
            let (q, qdot) = closed_state(seed, &params);
            let (a, _) = constrained_accelerations(&q, &qdot, &tau, &params, &gains).unwrap();
            let j = golfer::constraint_jacobian(&q, &params);
            let gamma = constraint_acceleration_bias(&q, &qdot, &params);
            let phi = golfer::constraint_vector(&q, &params);
            let qdot_vec = SVector::<f64, 8>::from_row_slice(&qdot);
            let rhs = -gamma - 2.0 * gains.alpha * (j * qdot_vec) - gains.beta * gains.beta * phi;
            let residual = j * a - rhs;
            for &row in INDEPENDENT_CLOSURE_ROWS.iter() {
                assert!(residual[row].abs() < 1e-10, "row {row}: {}", residual[row]);
            }
        }
    }

    #[test]
    fn rank_loss_on_independent_rows_is_a_typed_error() {
        let mut phi_q = SMatrix::<f64, 4, 8>::zeros();
        phi_q[(0, 0)] = 1.0;
        phi_q[(1, 0)] = 2.0; // rows 0-1 parallel
        let err = solve_independent_kkt(
            &SMatrix::<f64, 8, 8>::identity(),
            &phi_q,
            &SVector::<f64, 8>::zeros(),
            &SVector::<f64, 4>::zeros(),
        )
        .unwrap_err();
        assert!(matches!(err, ClosureError::RankLoss { .. }), "{err:?}");
        assert!(err.to_string().contains("rank"));
    }

    #[test]
    fn non_finite_input_is_a_typed_error() {
        let mut phi_q = SMatrix::<f64, 4, 8>::zeros();
        phi_q[(0, 0)] = 1.0;
        phi_q[(1, 1)] = 1.0;
        let mut rhs_dyn = SVector::<f64, 8>::zeros();
        rhs_dyn[0] = f64::NAN;
        let err = solve_independent_kkt(
            &SMatrix::<f64, 8, 8>::identity(),
            &phi_q,
            &rhs_dyn,
            &SVector::<f64, 4>::zeros(),
        )
        .unwrap_err();
        assert_eq!(err, ClosureError::NonFinite("rhs_dyn"));
    }

    #[test]
    fn redundant_rows_of_the_kkt_inputs_are_ignored() {
        let mut phi_q = SMatrix::<f64, 4, 8>::zeros();
        phi_q[(0, 0)] = 1.0;
        phi_q[(1, 1)] = 1.0;
        let base = solve_independent_kkt(
            &SMatrix::<f64, 8, 8>::identity(),
            &phi_q,
            &SVector::<f64, 8>::from_element(0.5),
            &SVector::<f64, 4>::new(0.3, -0.2, 0.0, 0.0),
        )
        .unwrap();
        phi_q[(2, 0)] = 7.0;
        phi_q[(3, 1)] = -3.0;
        let perturbed = solve_independent_kkt(
            &SMatrix::<f64, 8, 8>::identity(),
            &phi_q,
            &SVector::<f64, 8>::from_element(0.5),
            &SVector::<f64, 4>::new(0.3, -0.2, 9.0, 9.0),
        )
        .unwrap();
        assert_eq!(base, perturbed);
    }

    #[test]
    fn projections_close_the_loop_and_zero_the_closure_velocity() {
        let params = params();
        for seed in 0..8 {
            let (q, qdot) = closed_state(seed, &params);
            let phi = golfer::constraint_vector(&q, &params);
            assert!(phi.norm() < 1e-10, "phi = {phi:?}");
            let j = golfer::constraint_jacobian(&q, &params);
            let velocity = j * SVector::<f64, 8>::from_row_slice(&qdot);
            assert!(velocity.norm() < 1e-9, "J qdot = {velocity:?}");
        }
    }

    #[test]
    fn test_constrained_accelerations_kkt() {
        let params = params();
        let q = [0.0; 8];
        let qdot = [0.0; 8];
        let tau = [0.0; 8];
        let gains = BaumgarteGains::default();

        // The all-zero pose is open (Phi != 0) but not singular.
        let (a, _lambda) = constrained_accelerations(&q, &qdot, &tau, &params, &gains)
            .expect("zero pose has full-rank independent rows");

        for i in 0..8 {
            assert!(a[i].is_finite());
        }
    }
}
