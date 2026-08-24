//! Impulse, work, and power calculations for pendulum trajectory data.
//!
//! Operates on pre-computed trajectory arrays (torques, velocities, forces).
//! All functions are pure, stateless, and designed for bulk computation.
//!
//! Design by Contract:
//! - All input arrays must have matching lengths.
//! - All values must be finite.
//! - Time arrays must be strictly increasing.
//!
//! References:
//! Winter, D. A. (2009). Biomechanics and Motor Control of Human Movement.

/// Compute element-wise angular power: P[i] = torque[i] * omega[i].
///
/// # Panics
/// Panics if the slices have different lengths or contain non-finite values.
pub fn angular_power_series(torques: &[f64], angular_velocities: &[f64]) -> Vec<f64> {
    assert_eq!(
        torques.len(),
        angular_velocities.len(),
        "torques and angular_velocities must have the same length"
    );
    debug_assert!(
        torques.iter().all(|v| v.is_finite()),
        "torques must be all finite"
    );
    debug_assert!(
        angular_velocities.iter().all(|v| v.is_finite()),
        "angular_velocities must be all finite"
    );

    torques
        .iter()
        .zip(angular_velocities.iter())
        .map(|(t, w)| t * w)
        .collect()
}

/// Compute element-wise linear power: P[i] = F[i] · v[i].
///
/// Forces and velocities are flat arrays of length 2*N: [fx0, fy0, fx1, fy1, ...].
///
/// # Panics
/// Panics if lengths don't match or aren't multiples of 2.
pub fn linear_power_series(forces: &[f64], velocities: &[f64]) -> Vec<f64> {
    assert_eq!(
        forces.len(),
        velocities.len(),
        "forces and velocities must match"
    );
    assert_eq!(
        forces.len() % 2,
        0,
        "forces length must be even (2D vectors)"
    );

    forces
        .as_chunks::<2>()
        .0
        .iter()
        .zip(velocities.as_chunks::<2>().0.iter())
        .map(|(f, v)| f[0] * v[0] + f[1] * v[1])
        .collect()
}

/// Cumulative angular work via trapezoidal integration.
///
/// W[0] = 0, W[i] = W[i-1] + 0.5 * (P[i-1] + P[i]) * dt[i].
///
/// # Panics
/// Panics on mismatched lengths or non-increasing time.
pub fn angular_work_series(torques: &[f64], angular_velocities: &[f64], time: &[f64]) -> Vec<f64> {
    let power = angular_power_series(torques, angular_velocities);
    cumulative_trapz(&power, time)
}

/// Cumulative linear work via trapezoidal integration.
///
/// Forces and velocities are flat arrays of length 2*N.
pub fn linear_work_series(forces: &[f64], velocities: &[f64], time: &[f64]) -> Vec<f64> {
    let power = linear_power_series(forces, velocities);
    cumulative_trapz(&power, time)
}

/// Cumulative angular impulse via trapezoidal integration.
///
/// J[0] = 0, J[i] = J[i-1] + 0.5 * (tau[i-1] + tau[i]) * dt[i].
pub fn angular_impulse_series(torques: &[f64], time: &[f64]) -> Vec<f64> {
    assert_eq!(torques.len(), time.len(), "torques and time must match");
    cumulative_trapz(torques, time)
}

/// Cumulative linear impulse (2D) via trapezoidal integration.
///
/// Forces are flat: [fx0, fy0, fx1, fy1, ...].
/// Returns flat: [Jx0, Jy0, Jx1, Jy1, ...] with J[0] = (0, 0).
pub fn linear_impulse_series(forces: &[f64], time: &[f64]) -> Vec<f64> {
    assert_eq!(forces.len() % 2, 0, "forces length must be even");
    let n = forces.len() / 2;
    assert_eq!(n, time.len(), "N force pairs must match time length");

    let mut result = vec![0.0; forces.len()];
    if n > 1 {
        let mut acc_x = 0.0;
        let mut acc_y = 0.0;
        for i in 1..n {
            let dt = time[i] - time[i - 1];
            acc_x += 0.5 * (forces[2 * (i - 1)] + forces[2 * i]) * dt;
            acc_y += 0.5 * (forces[2 * (i - 1) + 1] + forces[2 * i + 1]) * dt;
            result[2 * i] = acc_x;
            result[2 * i + 1] = acc_y;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Shared trapezoidal integration
// ---------------------------------------------------------------------------

/// Cumulative trapezoidal integration: result[0] = 0, result[i] += 0.5*(f[i-1]+f[i])*dt.
fn cumulative_trapz(values: &[f64], time: &[f64]) -> Vec<f64> {
    assert_eq!(values.len(), time.len(), "values and time must match");
    let n = values.len();
    let mut result = vec![0.0; n];
    if n > 1 {
        let mut acc = 0.0;
        for i in 1..n {
            let dt = time[i] - time[i - 1];
            debug_assert!(dt > 0.0, "time must be strictly increasing at index {i}");
            acc += 0.5 * (values[i - 1] + values[i]) * dt;
            result[i] = acc;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_angular_power_basic() {
        let torques = [1.0, 2.0, 3.0];
        let omegas = [4.0, 5.0, 6.0];
        let p = angular_power_series(&torques, &omegas);
        assert_eq!(p, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_angular_power_zero() {
        let torques = [0.0, 0.0];
        let omegas = [5.0, 5.0];
        let p = angular_power_series(&torques, &omegas);
        assert_eq!(p, vec![0.0, 0.0]);
    }

    #[test]
    fn test_linear_power_basic() {
        // Two timesteps, 2D: forces = [(1,0), (0,1)], vels = [(3,4), (5,6)]
        let forces = [1.0, 0.0, 0.0, 1.0];
        let vels = [3.0, 4.0, 5.0, 6.0];
        let p = linear_power_series(&forces, &vels);
        assert_eq!(p.len(), 2);
        assert!(approx_eq(p[0], 3.0, 1e-12)); // 1*3 + 0*4
        assert!(approx_eq(p[1], 6.0, 1e-12)); // 0*5 + 1*6
    }

    #[test]
    fn test_angular_work_constant_power() {
        // Constant power of 2.0 over uniform time
        let torques = [2.0, 2.0, 2.0, 2.0];
        let omegas = [1.0, 1.0, 1.0, 1.0];
        let time = [0.0, 1.0, 2.0, 3.0];
        let w = angular_work_series(&torques, &omegas, &time);

        assert!(approx_eq(w[0], 0.0, 1e-12));
        assert!(approx_eq(w[1], 2.0, 1e-12));
        assert!(approx_eq(w[2], 4.0, 1e-12));
        assert!(approx_eq(w[3], 6.0, 1e-12));
    }

    #[test]
    fn test_angular_impulse_constant_torque() {
        let torques = [3.0, 3.0, 3.0];
        let time = [0.0, 1.0, 2.0];
        let j = angular_impulse_series(&torques, &time);

        assert!(approx_eq(j[0], 0.0, 1e-12));
        assert!(approx_eq(j[1], 3.0, 1e-12)); // 0.5*(3+3)*1
        assert!(approx_eq(j[2], 6.0, 1e-12)); // cumulative
    }

    #[test]
    fn test_linear_impulse_basic() {
        let forces = [1.0, 2.0, 3.0, 4.0]; // 2 timesteps
        let time = [0.0, 1.0];
        let j = linear_impulse_series(&forces, &time);

        assert!(approx_eq(j[0], 0.0, 1e-12));
        assert!(approx_eq(j[1], 0.0, 1e-12));
        assert!(approx_eq(j[2], 0.5 * (1.0 + 3.0), 1e-12)); // Jx
        assert!(approx_eq(j[3], 0.5 * (2.0 + 4.0), 1e-12)); // Jy
    }

    #[test]
    fn test_cumulative_trapz_linear_ramp() {
        // Linearly increasing: f = [0, 1, 2, 3], dt = 1
        // Integral of linear ramp: 0.5, 2.0, 4.5
        let values = [0.0, 1.0, 2.0, 3.0];
        let time = [0.0, 1.0, 2.0, 3.0];
        let result = cumulative_trapz(&values, &time);

        assert!(approx_eq(result[0], 0.0, 1e-12));
        assert!(approx_eq(result[1], 0.5, 1e-12));
        assert!(approx_eq(result[2], 2.0, 1e-12));
        assert!(approx_eq(result[3], 4.5, 1e-12));
    }

    #[test]
    fn test_empty_series() {
        let p = angular_power_series(&[], &[]);
        assert!(p.is_empty());
    }

    #[test]
    fn test_single_element_work() {
        let w = angular_work_series(&[5.0], &[3.0], &[0.0]);
        assert_eq!(w.len(), 1);
        assert!(approx_eq(w[0], 0.0, 1e-12));
    }

    #[test]
    #[should_panic(expected = "torques and angular_velocities must have the same length")]
    fn test_mismatched_lengths_panics() {
        angular_power_series(&[1.0, 2.0], &[1.0]);
    }
}
