//! Criterion benchmarks for math-primitives.
//!
//! Measures per-call latency for rotation, quaternion, and geometry operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use math_primitives::rotation::{euler_to_rotation_matrix, rotation_matrix_to_euler};
use math_primitives::quaternion::Quaternion;
use math_primitives::types::Vector3;

fn bench_euler_to_rotmat(c: &mut Criterion) {
    let euler = [0.1, 0.2, 0.3];
    c.bench_function("euler_to_rotation_matrix", |b| {
        b.iter(|| euler_to_rotation_matrix(black_box(&euler)))
    });
}

fn bench_rotmat_to_euler(c: &mut Criterion) {
    let rotmat = euler_to_rotation_matrix(&[0.1, 0.2, 0.3]);
    c.bench_function("rotation_matrix_to_euler", |b| {
        b.iter(|| rotation_matrix_to_euler(black_box(&rotmat)))
    });
}

fn bench_quaternion_multiply(c: &mut Criterion) {
    let q1 = Quaternion::new(1.0, 0.0, 0.0, 0.0).unwrap();
    let q2 = Quaternion::new(0.707, 0.707, 0.0, 0.0).unwrap();
    c.bench_function("quaternion_multiply", |b| {
        b.iter(|| black_box(&q1).multiply(black_box(&q2)))
    });
}

fn bench_quaternion_from_axis_angle(c: &mut Criterion) {
    let axis = Vector3::new(0.0, 0.0, 1.0);
    c.bench_function("quaternion_from_axis_angle", |b| {
        b.iter(|| Quaternion::from_axis_angle(black_box(&axis), black_box(0.5)))
    });
}

fn bench_slerp(c: &mut Criterion) {
    let q1 = Quaternion::new(1.0, 0.0, 0.0, 0.0).unwrap();
    let q2 = Quaternion::new(0.707, 0.707, 0.0, 0.0).unwrap();
    c.bench_function("slerp_midpoint", |b| {
        b.iter(|| black_box(&q1).slerp(black_box(&q2), black_box(0.5)))
    });
}

criterion_group!(
    benches,
    bench_euler_to_rotmat,
    bench_rotmat_to_euler,
    bench_quaternion_multiply,
    bench_quaternion_from_axis_angle,
    bench_slerp,
);
criterion_main!(benches);
