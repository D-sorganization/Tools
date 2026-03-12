//! Batch simulation and parallel evaluation for population-based optimization.
//!
//! Uses rayon for embarrassingly parallel evaluation of candidate torque profiles.

use crate::double::{coriolis, forward_kinematics, gravity_vector, mass_matrix};
use crate::integrator::{integrate_rk45, RK45Config};
use crate::types::DoublePendulumParams;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Result of a single batch simulation evaluation.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Maximum tip speed achieved during the simulation (m/s)
    pub max_tip_speed: f64,
    /// Tip speed at the bottom of the swing arc (m/s)
    pub tip_speed_at_bottom: f64,
    /// Final time of the simulation
    pub t_final: f64,
    /// Whether the simulation completed successfully
    pub success: bool,
}

/// Evaluate a polynomial torque profile on the double pendulum.
///
/// # Arguments
/// * `params` - Physical parameters of the double pendulum
/// * `coeffs` - Polynomial coefficients for torque1 and torque2 (interleaved:
///   [t1_c0, t1_c1, ..., t2_c0, t2_c1, ...])
/// * `n_coeffs_per_joint` - Number of polynomial coefficients per joint
/// * `q0` - Initial joint angles [theta1, phi]
/// * `qdot0` - Initial joint velocities [dtheta1, dphi]
/// * `t_end` - Simulation end time (s)
fn evaluate_single(
    params: &DoublePendulumParams,
    coeffs: &[f64],
    n_coeffs_per_joint: usize,
    q0: [f64; 2],
    qdot0: [f64; 2],
    t_end: f64,
) -> BatchResult {
    let (coeffs1, coeffs2) = coeffs.split_at(n_coeffs_per_joint);

    // Build the ODE right-hand side
    let config = RK45Config {
        h0: 0.005,
        h_min: 1e-6,
        h_max: 0.01,
        rtol: 1e-6,
        atol: 1e-9,
        max_steps: 50000,
    };

    let mut y0 = [0.0; 4];
    y0[0] = q0[0];
    y0[1] = q0[1];
    y0[2] = qdot0[0];
    y0[3] = qdot0[1];

    let result = integrate_rk45(
        |t, y| {
            let q = [y[0], y[1]];
            let qdot_val = [y[2], y[3]];

            // Evaluate polynomial torques
            let tau1: f64 = coeffs1
                .iter()
                .enumerate()
                .map(|(i, c)| c * t.powi(i as i32))
                .sum();
            let tau2: f64 = coeffs2
                .iter()
                .enumerate()
                .map(|(i, c)| c * t.powi(i as i32))
                .sum();

            // Compute dynamics: M * qddot = tau - C - G
            let m = mass_matrix(&q, params);
            let c = coriolis(&q, &qdot_val, params);
            let g = gravity_vector(&q, params);

            let rhs = nalgebra::SVector::<f64, 2>::new(tau1, tau2) - c - g;

            // Solve M * qddot = rhs
            let lu = m.lu();
            let qddot = match lu.solve(&rhs) {
                Some(sol) => sol,
                None => return [qdot_val[0], qdot_val[1], 0.0, 0.0],
            };

            [qdot_val[0], qdot_val[1], qddot[0], qddot[1]]
        },
        0.0,
        t_end,
        y0,
        config,
    );

    // Compute tip speeds
    let mut max_tip_speed: f64 = 0.0;
    let mut tip_speed_bottom: f64 = 0.0;
    let mut min_tip_y = f64::INFINITY;

    for step in &result {
        let q = [step.y[0], step.y[1]];
        let qdot_val = [step.y[2], step.y[3]];

        let fk = forward_kinematics(&q, params);
        let abs2 = q[0] + q[1];
        let dabs2 = qdot_val[0] + qdot_val[1];

        let vwx = params.l1 * q[0].cos() * qdot_val[0];
        let vwy = params.l1 * q[0].sin() * qdot_val[0];
        let vtx = vwx + params.l2 * abs2.cos() * dabs2;
        let vty = vwy + params.l2 * abs2.sin() * dabs2;

        let tip_speed = (vtx * vtx + vty * vty).sqrt();
        if tip_speed > max_tip_speed {
            max_tip_speed = tip_speed;
        }

        // Track the point closest to the bottom
        if fk.club_tip.1 < min_tip_y {
            min_tip_y = fk.club_tip.1;
            tip_speed_bottom = tip_speed;
        }
    }

    let success = result.len() > 1 && max_tip_speed.is_finite();

    BatchResult {
        max_tip_speed,
        tip_speed_at_bottom: tip_speed_bottom,
        t_final: result.last().map_or(0.0, |s| s.t),
        success,
    }
}

/// Evaluate a batch of polynomial torque profiles in parallel.
///
/// Each row of `coeffs_batch` is one candidate's coefficients.
/// Returns a vector of `BatchResult`s, one per candidate.
#[cfg(feature = "parallel")]
pub fn batch_evaluate_double(
    params: &DoublePendulumParams,
    coeffs_batch: &[Vec<f64>],
    n_coeffs_per_joint: usize,
    q0: [f64; 2],
    qdot0: [f64; 2],
    t_end: f64,
) -> Vec<BatchResult> {
    coeffs_batch
        .par_iter()
        .map(|coeffs| evaluate_single(params, coeffs, n_coeffs_per_joint, q0, qdot0, t_end))
        .collect()
}

/// Evaluate a batch of polynomial torque profiles sequentially (fallback).
#[cfg(not(feature = "parallel"))]
pub fn batch_evaluate_double(
    params: &DoublePendulumParams,
    coeffs_batch: &[Vec<f64>],
    n_coeffs_per_joint: usize,
    q0: [f64; 2],
    qdot0: [f64; 2],
    t_end: f64,
) -> Vec<BatchResult> {
    coeffs_batch
        .iter()
        .map(|coeffs| evaluate_single(params, coeffs, n_coeffs_per_joint, q0, qdot0, t_end))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_evaluation() {
        let params = DoublePendulumParams {
            m1: 5.0,
            m2: 0.3,
            m_clubhead: 0.2,
            l1: 0.65,
            l2: 1.10,
            g: 9.81,
            friction1: 0.0,
            friction2: 0.0,
        };

        let coeffs = vec![10.0, -5.0, 0.0, 5.0, -2.0, 0.0]; // 3 per joint
        let result = evaluate_single(
            &params,
            &coeffs,
            3,
            [std::f64::consts::FRAC_PI_2, 0.0],
            [0.0, 0.0],
            1.0,
        );

        assert!(result.success);
        assert!(result.max_tip_speed > 0.0);
    }

    #[test]
    fn test_batch_evaluation() {
        let params = DoublePendulumParams {
            m1: 5.0,
            m2: 0.3,
            m_clubhead: 0.2,
            l1: 0.65,
            l2: 1.10,
            g: 9.81,
            friction1: 0.0,
            friction2: 0.0,
        };

        let batch = vec![
            vec![10.0, 0.0, 5.0, 0.0],
            vec![20.0, 0.0, 10.0, 0.0],
            vec![5.0, 0.0, 2.0, 0.0],
        ];

        let results = batch_evaluate_double(
            &params,
            &batch,
            2,
            [std::f64::consts::FRAC_PI_2, 0.0],
            [0.0, 0.0],
            0.5,
        );

        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.success);
        }
    }
}
