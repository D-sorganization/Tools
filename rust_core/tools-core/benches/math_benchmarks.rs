//! Benchmarks for math primitives using Criterion.
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tools_core::ball_flight::{
    self, analyze_trajectory, apply_spin_decay, calculate_accel_core, simulate_trajectory,
    BallProperties, EnvironmentalConditions, LaunchConditions,
};
use tools_core::math;
use tools_core::matrix3::Matrix3;
use tools_core::quaternion::Quaternion;
use tools_core::types::Vector3;

fn bench_vector3_magnitude(c: &mut Criterion) {
    let v = Vector3::new(3.0, 4.0, 5.0);
    c.bench_function("Vector3::magnitude", |b| {
        b.iter(|| black_box(v).magnitude())
    });
}

fn bench_vector3_cross(c: &mut Criterion) {
    let a = Vector3::new(1.0, 2.0, 3.0);
    let b = Vector3::new(4.0, 5.0, 6.0);
    c.bench_function("Vector3::cross", |b_iter| {
        b_iter.iter(|| black_box(a).cross(black_box(&b)))
    });
}

fn bench_vector3_normalize(c: &mut Criterion) {
    let v = Vector3::new(3.0, 4.0, 5.0);
    c.bench_function("Vector3::normalized", |b| {
        b.iter(|| black_box(v).normalized())
    });
}

fn bench_quaternion_multiply(c: &mut Criterion) {
    let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0).unwrap();
    let q2 = Quaternion::new(4.0, 3.0, 2.0, 1.0).unwrap();
    c.bench_function("Quaternion::multiply", |b| {
        b.iter(|| black_box(q1).multiply(black_box(&q2)))
    });
}

fn bench_quaternion_rotate_vector(c: &mut Criterion) {
    let axis = Vector3::new(0.0, 0.0, 1.0);
    let q = Quaternion::from_axis_angle(&axis, 0.5).unwrap();
    let v = Vector3::new(1.0, 0.0, 0.0);
    c.bench_function("Quaternion::rotate_vector", |b| {
        b.iter(|| black_box(q).rotate_vector(black_box(&v)))
    });
}

fn bench_quaternion_slerp(c: &mut Criterion) {
    let q1 = Quaternion::identity();
    let axis = Vector3::new(0.0, 0.0, 1.0);
    let q2 = Quaternion::from_axis_angle(&axis, 1.5).unwrap();
    c.bench_function("Quaternion::slerp", |b| {
        b.iter(|| black_box(q1).slerp(black_box(&q2), black_box(0.5)))
    });
}

fn bench_matrix3_mul_vec(c: &mut Criterion) {
    let axis = Vector3::new(1.0, 1.0, 1.0);
    let q = Quaternion::from_axis_angle(&axis, 1.0).unwrap();
    let m = Matrix3::from_quaternion(&q);
    let v = Vector3::new(1.0, 2.0, 3.0);
    c.bench_function("Matrix3::mul_vec", |b| {
        b.iter(|| black_box(m).mul_vec(black_box(&v)))
    });
}

fn bench_matrix3_mul_mat(c: &mut Criterion) {
    let m = Matrix3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    c.bench_function("Matrix3::mul_mat", |b| {
        b.iter(|| black_box(m).mul_mat(black_box(&m)))
    });
}

fn bench_lerp(c: &mut Criterion) {
    c.bench_function("lerp", |b| {
        b.iter(|| math::lerp(black_box(0.0), black_box(100.0), black_box(0.5)))
    });
}

// ── Ball Flight Physics Benchmarks ──────────────────────────────────────────

fn bench_accel_core(c: &mut Criterion) {
    let rel_vel = Vector3::new(68.0, 14.0, 0.0);
    let gravity = Vector3::new(0.0, -ball_flight::GRAVITY, 0.0);
    let coeffs = (0.21, 0.05, 0.02, 0.0, 0.38, 0.0);
    let spin_axis = Vector3::new(0.0, 0.0, 1.0);
    c.bench_function("calculate_accel_core", |b| {
        b.iter(|| {
            calculate_accel_core(
                black_box(&rel_vel),
                black_box(69.4),
                black_box(&gravity),
                black_box(0.021335),
                black_box(19.1),
                black_box(&coeffs),
                black_box(261.8),
                black_box(&spin_axis),
            )
        })
    });
}

fn bench_spin_decay(c: &mut Criterion) {
    c.bench_function("apply_spin_decay", |b| {
        b.iter(|| apply_spin_decay(black_box(261.8), black_box(0.08), black_box(0.01)))
    });
}

fn bench_simulate_trajectory(c: &mut Criterion) {
    let ball = BallProperties::default();
    let env = EnvironmentalConditions::default();
    let launch = LaunchConditions::default();
    c.bench_function("simulate_trajectory (dt=0.01)", |b| {
        b.iter(|| {
            simulate_trajectory(
                black_box(&ball),
                black_box(&env),
                black_box(&launch),
                10.0,
                0.01,
            )
        })
    });
}

fn bench_simulate_trajectory_fine(c: &mut Criterion) {
    let ball = BallProperties::default();
    let env = EnvironmentalConditions::default();
    let launch = LaunchConditions::default();
    c.bench_function("simulate_trajectory (dt=0.001)", |b| {
        b.iter(|| {
            simulate_trajectory(
                black_box(&ball),
                black_box(&env),
                black_box(&launch),
                10.0,
                0.001,
            )
        })
    });
}

fn bench_analyze_trajectory(c: &mut Criterion) {
    let ball = BallProperties::default();
    let env = EnvironmentalConditions::default();
    let launch = LaunchConditions::default();
    let traj = simulate_trajectory(&ball, &env, &launch, 10.0, 0.01);
    c.bench_function("analyze_trajectory", |b| {
        b.iter(|| analyze_trajectory(black_box(&traj)))
    });
}

criterion_group!(
    benches,
    bench_vector3_magnitude,
    bench_vector3_cross,
    bench_vector3_normalize,
    bench_quaternion_multiply,
    bench_quaternion_rotate_vector,
    bench_quaternion_slerp,
    bench_matrix3_mul_vec,
    bench_matrix3_mul_mat,
    bench_lerp,
    bench_accel_core,
    bench_spin_decay,
    bench_simulate_trajectory,
    bench_simulate_trajectory_fine,
    bench_analyze_trajectory,
);
criterion_main!(benches);
