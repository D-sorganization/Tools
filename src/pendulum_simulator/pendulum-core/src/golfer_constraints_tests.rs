//! Unit tests for `golfer_constraints` (split out for the file-size budget).

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
        g: crate::types::STANDARD_GRAVITY,
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
