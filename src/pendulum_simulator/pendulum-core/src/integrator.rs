//! Runge-Kutta 45 (Dormand-Prince) adaptive step ODE integrator.

/// Configuration for the RK45 integrator.
#[derive(Debug, Clone, Copy)]
pub struct RK45Config {
    /// Initial step size
    pub h0: f64,
    /// Minimum step size
    pub h_min: f64,
    /// Maximum step size
    pub h_max: f64,
    /// Relative error tolerance
    pub rtol: f64,
    /// Absolute error tolerance
    pub atol: f64,
    /// Maximum number of steps
    pub max_steps: usize,
}

impl Default for RK45Config {
    fn default() -> Self {
        RK45Config {
            h0: 0.01,
            h_min: 1e-6,
            h_max: 0.1,
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 100000,
        }
    }
}

/// Result of a single integration step.
#[derive(Debug, Clone)]
pub struct IntegrationStep<const N: usize> {
    /// Time at this step
    pub t: f64,
    /// State vector at this step
    pub y: [f64; N],
    /// Step size used
    pub h: f64,
}

/// Dorman-Prince RK45 coefficients (7-stage method).
/// This computes a 5th-order accurate estimate with 4th-order local error control.
const RK45_A: &[&[f64]] = &[
    &[],
    &[1.0 / 5.0],
    &[3.0 / 40.0, 9.0 / 40.0],
    &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    &[
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    ],
    &[
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ],
    &[
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];

const RK45_B: &[f64] = &[
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];

const RK45_B_STAR: &[f64] = &[
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
];

const RK45_C: &[f64] = &[0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

/// Generic RK45 integrator for n-dimensional ODE systems.
///
/// Solves: dy/dt = f(t, y) with adaptive step control.
pub fn integrate_rk45<F, const N: usize>(
    f: F,
    t0: f64,
    t_end: f64,
    y0: [f64; N],
    config: RK45Config,
) -> Vec<IntegrationStep<N>>
where
    F: Fn(f64, &[f64; N]) -> [f64; N],
{
    let mut result = vec![IntegrationStep {
        t: t0,
        y: y0,
        h: config.h0,
    }];

    let mut t = t0;
    let mut y = y0;
    let mut h = config.h0;

    let mut step_count = 0;

    while t < t_end && step_count < config.max_steps {
        // Adjust step size to not overshoot
        if t + h > t_end {
            h = t_end - t;
        }

        // Compute RK45 stages
        let mut k = [[0.0; N]; 7];

        k[0] = f(t, &y);

        for i in 1..7 {
            let mut y_stage = y;
            for j in 0..N {
                for l in 0..i {
                    y_stage[j] += h * RK45_A[i][l] * k[l][j];
                }
            }
            k[i] = f(t + h * RK45_C[i], &y_stage);
        }

        // Compute 5th-order solution and 4th-order solution
        let mut y5 = [0.0; N];
        let mut y4 = [0.0; N];

        for j in 0..N {
            y5[j] = y[j];
            y4[j] = y[j];

            for i in 0..7 {
                y5[j] += h * RK45_B[i] * k[i][j];
                y4[j] += h * RK45_B_STAR[i] * k[i][j];
            }
        }

        // Compute error estimate
        let mut error: f64 = 0.0;
        for j in 0..N {
            let tol = config.atol + config.rtol * (y5[j].abs().max(y[j].abs()));
            let err_term = (y5[j] - y4[j]) / tol;
            error = error.max(err_term.abs());
        }

        // Step acceptance and size control
        let q: f64 = 0.84 * (1.0 / (error + 1e-10)).powf(0.25);
        let h_new = h * q.clamp(0.1, 4.0);

        if error <= 1.0 {
            // Accept step
            t += h;
            y = y5;

            result.push(IntegrationStep { t, y, h });

            step_count += 1;
        }

        h = h_new.clamp(config.h_min, config.h_max);
    }

    result
}

/// Integrate a 2-DOF system (double pendulum).
pub fn integrate_double_pendulum<F>(
    f: F,
    t0: f64,
    t_end: f64,
    q0: [f64; 2],
    qdot0: [f64; 2],
    config: RK45Config,
) -> Vec<IntegrationStep<4>>
where
    F: Fn(f64, &[f64; 2], &[f64; 2]) -> [f64; 2],
{
    let mut y0 = [0.0; 4];
    y0[0] = q0[0];
    y0[1] = q0[1];
    y0[2] = qdot0[0];
    y0[3] = qdot0[1];

    integrate_rk45(
        |t, y| {
            let q = [y[0], y[1]];
            let qdot = [y[2], y[3]];
            let qddot = f(t, &q, &qdot);
            [qdot[0], qdot[1], qddot[0], qddot[1]]
        },
        t0,
        t_end,
        y0,
        config,
    )
}

/// Integrate a 3-DOF system (triple pendulum).
pub fn integrate_triple_pendulum<F>(
    f: F,
    t0: f64,
    t_end: f64,
    q0: [f64; 3],
    qdot0: [f64; 3],
    config: RK45Config,
) -> Vec<IntegrationStep<6>>
where
    F: Fn(f64, &[f64; 3], &[f64; 3]) -> [f64; 3],
{
    let mut y0 = [0.0; 6];
    y0[0] = q0[0];
    y0[1] = q0[1];
    y0[2] = q0[2];
    y0[3] = qdot0[0];
    y0[4] = qdot0[1];
    y0[5] = qdot0[2];

    integrate_rk45(
        |t, y| {
            let q = [y[0], y[1], y[2]];
            let qdot = [y[3], y[4], y[5]];
            let qddot = f(t, &q, &qdot);
            [qdot[0], qdot[1], qdot[2], qddot[0], qddot[1], qddot[2]]
        },
        t0,
        t_end,
        y0,
        config,
    )
}

/// Integrate an 8-DOF system (golfer).
pub fn integrate_golfer<F>(
    f: F,
    t0: f64,
    t_end: f64,
    q0: [f64; 8],
    qdot0: [f64; 8],
    config: RK45Config,
) -> Vec<IntegrationStep<16>>
where
    F: Fn(f64, &[f64; 8], &[f64; 8]) -> [f64; 8],
{
    let mut y0 = [0.0; 16];
    y0[..8].copy_from_slice(&q0);
    y0[8..16].copy_from_slice(&qdot0);

    integrate_rk45(
        |t, y| {
            let q = [y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]];
            let qdot = [y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]];
            let qddot = f(t, &q, &qdot);
            [
                qdot[0], qdot[1], qdot[2], qdot[3], qdot[4], qdot[5], qdot[6], qdot[7], qddot[0],
                qddot[1], qddot[2], qddot[3], qddot[4], qddot[5], qddot[6], qddot[7],
            ]
        },
        t0,
        t_end,
        y0,
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rk45_simple_ode() {
        // Test dy/dt = -y, y(0) = 1, solution y(t) = e^{-t}
        let f = |_t: f64, y: &[f64; 1]| [-y[0]];

        let config = RK45Config {
            h0: 0.01,
            h_min: 1e-6,
            h_max: 0.1,
            rtol: 1e-4,
            atol: 1e-7,
            max_steps: 10000,
        };

        let result = integrate_rk45(f, 0.0, 1.0, [1.0], config);

        // At t=1, y should be approximately e^{-1} ≈ 0.3679
        let final_y = result.last().unwrap().y[0];
        assert!((final_y - (-1.0_f64).exp()).abs() < 1e-3);
    }

    #[test]
    fn test_rk45_steps_are_ordered() {
        let f = |_t: f64, y: &[f64; 1]| [-y[0]];

        let config = RK45Config::default();
        let result = integrate_rk45(f, 0.0, 1.0, [1.0], config);

        // Times should be strictly increasing
        for i in 1..result.len() {
            assert!(result[i].t > result[i - 1].t);
        }

        // First time should be t0, last should be close to t_end
        assert_eq!(result[0].t, 0.0);
        assert!(result.last().unwrap().t >= 0.99);
    }
}
