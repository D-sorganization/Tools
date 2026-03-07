//! Benchmarks for math primitives using Criterion.
//!
//! Run with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tools_core::math;
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

fn bench_lerp(c: &mut Criterion) {
    c.bench_function("lerp", |b| {
        b.iter(|| math::lerp(black_box(0.0), black_box(100.0), black_box(0.5)))
    });
}

fn bench_deg_to_rad(c: &mut Criterion) {
    c.bench_function("deg_to_rad", |b| {
        b.iter(|| math::deg_to_rad(black_box(45.0)))
    });
}

criterion_group!(
    benches,
    bench_vector3_magnitude,
    bench_vector3_cross,
    bench_vector3_normalize,
    bench_lerp,
    bench_deg_to_rad,
);
criterion_main!(benches);
