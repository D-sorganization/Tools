//! Criterion benchmarks for pendulum-core physics kernels.
//!
//! Measures per-call latency for the most performance-critical functions:
//! - Mass matrix computation
//! - Forward kinematics
//! - Constrained dynamics (KKT solve)
//! - RK45 integration (full simulation)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pendulum_core::*;

fn default_double_params() -> DoublePendulumParams {
    DoublePendulumParams {
        m1: 5.0,
        m2: 3.0,
        m_clubhead: 0.3,
        l1: 0.7,
        l2: 1.1,
        g: 9.81,
        friction1: 0.01,
        friction2: 0.01,
    }
}

fn default_triple_params() -> TriplePendulumParams {
    TriplePendulumParams {
        masses: [5.0, 3.0, 2.0],
        lengths: [0.5, 0.4, 1.1],
        g: 9.81,
        friction: [0.01, 0.01, 0.01],
    }
}

fn bench_double_mass_matrix(c: &mut Criterion) {
    let params = default_double_params();
    let q = [0.3, -0.5];
    c.bench_function("double_mass_matrix", |b| {
        b.iter(|| double_mass_matrix(black_box(&q), black_box(&params)))
    });
}

fn bench_double_forward_kinematics(c: &mut Criterion) {
    let params = default_double_params();
    let q = [0.3, -0.5];
    c.bench_function("double_forward_kinematics", |b| {
        b.iter(|| double_forward_kinematics(black_box(&q), black_box(&params)))
    });
}

fn bench_double_eom(c: &mut Criterion) {
    let params = default_double_params();
    let q = [0.3, -0.5];
    let qdot = [0.1, -0.2];
    let tau = [10.0, -5.0];
    c.bench_function("double_equations_of_motion", |b| {
        b.iter(|| {
            double_equations_of_motion(
                black_box(&q),
                black_box(&qdot),
                black_box(&tau),
                black_box(&params),
            )
        })
    });
}

fn bench_triple_mass_matrix(c: &mut Criterion) {
    let params = default_triple_params();
    let q = [0.3, -0.5, 0.2];
    c.bench_function("triple_mass_matrix", |b| {
        b.iter(|| triple_mass_matrix(black_box(&q), black_box(&params)))
    });
}

fn bench_triple_eom(c: &mut Criterion) {
    let params = default_triple_params();
    let q = [0.3, -0.5, 0.2];
    let qdot = [0.1, -0.2, 0.15];
    let tau = [10.0, -5.0, 3.0];
    c.bench_function("triple_equations_of_motion", |b| {
        b.iter(|| {
            triple_equations_of_motion(
                black_box(&q),
                black_box(&qdot),
                black_box(&tau),
                black_box(&params),
            )
        })
    });
}

fn bench_rk45_double_integration(c: &mut Criterion) {
    let params = default_double_params();
    let config = RK45Config {
        h0: 0.005,
        h_min: 1e-6,
        h_max: 0.01,
        rtol: 1e-6,
        atol: 1e-9,
        max_steps: 100_000,
    };

    c.bench_function("rk45_double_1s", |b| {
        b.iter(|| {
            integrator::integrate_rk45(
                |t, y| {
                    let q = [y[0], y[1]];
                    let qd = [y[2], y[3]];
                    let tau1 = 50.0 * (1.0 - t);
                    let tau2 = -30.0 * t;
                    let qddot =
                        double_equations_of_motion(&q, &qd, &[tau1, tau2], &params);
                    [qd[0], qd[1], qddot[0], qddot[1]]
                },
                0.0,
                black_box(1.0),
                [0.0, 0.0, 0.0, 0.0],
                config,
            )
        })
    });
}

criterion_group!(
    benches,
    bench_double_mass_matrix,
    bench_double_forward_kinematics,
    bench_double_eom,
    bench_triple_mass_matrix,
    bench_triple_eom,
    bench_rk45_double_integration,
);
criterion_main!(benches);
