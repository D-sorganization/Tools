//! Criterion coverage for the compiled flight-to-ground reference path.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tools_core::flight_ground::{
    parse_ground_reference_execution_v1_json, parse_request_v1_json, run_ground_reference_v1,
};

const FIXTURE: &str = include_str!(
    "../../../src/rate_of_closure/web/src/model/__fixtures__/ground_reference_pipeline_golden_v1.json"
);

fn ground_reference(c: &mut Criterion) {
    let value: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    let mut execution = value["execution"].clone();
    execution["schema_version"] = value["execution_schema_version"].clone();
    let request = parse_request_v1_json(&value["request"].to_string()).expect("valid request");
    let execution =
        parse_ground_reference_execution_v1_json(&execution.to_string()).expect("valid execution");

    c.bench_function("ground_reference/canonical_fixture", |b| {
        b.iter(|| {
            black_box(
                run_ground_reference_v1(black_box(&request), black_box(&execution), || false)
                    .expect("qualified fixture"),
            )
        });
    });
}

criterion_group!(benches, ground_reference);
criterion_main!(benches);
