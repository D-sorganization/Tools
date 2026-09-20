"""Tests for the flight validation package and benchmark carry evaluation (#4252)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.python.swing_sim.flight.validation import (
    BENCHMARK_LAUNCHES,
    VALIDATION_SCHEMA_VERSION,
    evaluate_flight_benchmarks,
)


@pytest.mark.unit
def test_benchmark_launches_are_defined() -> None:
    assert len(BENCHMARK_LAUNCHES) == 5
    ids = [b.condition_id for b in BENCHMARK_LAUNCHES]
    assert ids == [
        "tour_driver",
        "amateur_driver",
        "mid_iron_5iron",
        "wedge",
        "high_speed_driver",
    ]
    for b in BENCHMARK_LAUNCHES:
        assert b.ball_speed_mps > 0.0
        assert b.launch_angle_deg > 0.0
        assert b.spin_rate_rpm > 0.0
        assert b.reference_carry_m > 0.0
        assert b.source_citation != ""
        assert b.license_status != ""


@pytest.mark.physics
@pytest.mark.unit
def test_evaluate_flight_benchmarks_generates_valid_report() -> None:
    report = evaluate_flight_benchmarks()
    assert report.schema_version == VALIDATION_SCHEMA_VERSION
    assert report.canonical_model == "Waterloo/Penner"

    # All 7 literature models evaluated
    assert len(report.model_evaluations) >= 7
    for _model_name, evals in report.model_evaluations.items():
        assert len(evals) == 5
        for _cond_id, cond_eval in evals.items():
            assert cond_eval.carry_m > 50.0
            assert cond_eval.max_height_m > 5.0
            assert cond_eval.flight_time_s > 1.0

    # Aggregate metrics reasonable across conditions
    assert "Waterloo/Penner" in report.aggregate_metrics
    waterloo_agg = report.aggregate_metrics["Waterloo/Penner"]
    assert waterloo_agg.mae_m < 3.0
    assert waterloo_agg.max_relative_error_pct < 5.0


@pytest.mark.parity
@pytest.mark.physics
def test_runtime_parity_all_conditions_within_one_percent() -> None:
    report = evaluate_flight_benchmarks()
    assert len(report.runtime_parity_checks) == 5
    for check in report.runtime_parity_checks:
        assert check.passed is True
        assert check.relative_diff_pct <= 1.0, (
            f"Parity failure for {check.condition_id}: {check.relative_diff_pct}%"
        )


@pytest.mark.unit
def test_validation_manifest_json_parity() -> None:
    json_path = (
        Path(__file__).parents[6]
        / "docs"
        / "development"
        / "flight_model_validation.json"
    )
    assert json_path.exists(), f"Missing {json_path}"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == VALIDATION_SCHEMA_VERSION
    assert data["canonical_model"] == "Waterloo/Penner"
    assert len(data["benchmark_launches"]) == 5
    assert len(data["limitations"]) == 4


@pytest.mark.unit
def test_limitations_are_documented() -> None:
    report = evaluate_flight_benchmarks()
    assert len(report.limitations) == 4
    joined = " ".join(report.limitations)
    assert "Terminal ground contact" in joined
    assert "Steady meteorological wind" in joined
    assert "MAX_GOLF_BALL_LIFT_COEFFICIENT" in joined
    assert "Spin decay" in joined
