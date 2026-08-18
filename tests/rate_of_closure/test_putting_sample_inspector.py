"""Contracts for the bounded, synchronized putting sample inspector."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from rate_of_closure.putting_result_contract import validate_putting_result_summary
from rate_of_closure.putting_sample_inspector import (
    MAX_PUTTING_DISPLAY_SAMPLES,
    PuttingSampleSeries,
    navigate_putting_samples,
    nearest_putting_sample,
    plan_putting_samples,
)
from shared.python.swing_sim.putting import GreenConditions, PuttLaunch, simulate_putt

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _golden() -> dict[str, object]:
    path = (
        Path(__file__).parents[2]
        / "src/rate_of_closure/web/src/model/__fixtures__"
        / "putting_sample_inspector_golden_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _series(source: dict[str, object]) -> PuttingSampleSeries:
    return PuttingSampleSeries(
        path_x_m=source["path_x_m"],
        path_y_m=source["path_y_m"],
        speeds_mps=source["speeds_mps"],
        times_s=source["times_s"],
        skid_end_index=source["skid_end_index"],
    )


def test_python_owned_golden_pins_geometry_phase_navigation_and_pixel_tie() -> None:
    fixture = _golden()
    source = fixture["series"]
    expected = fixture["expected"]
    assert isinstance(source, dict) and isinstance(expected, dict)
    tie_plan = plan_putting_samples(
        PuttingSampleSeries(
            tuple(float(i) for i in range(8)),
            (0.0,) * 8,
            tuple(float(8 - i) for i in range(8)),
            tuple(float(i) for i in range(8)),
            0,
        ),
        cap=5,
    )
    assert (
        list(tie_plan.displayed_raw_indices)
        == expected["half_tie_displayed_raw_indices_at_cap_5"]
    )
    plan = plan_putting_samples(_series(source), cap=6)

    assert [sample.raw_index for sample in plan.samples] == expected[
        "displayed_raw_indices_at_cap_6"
    ]
    assert plan.raw_count == 7
    assert plan.displayed_count == 6
    assert plan.skid_end_index == 3
    assert plan.cumulative_distance_m == pytest.approx(
        expected["cumulative_distance_m"]
    )
    assert [
        plan.raw_sample(index).phase for index in range(plan.raw_count)
    ] == expected["phases"]

    navigation = expected["navigation_from_3"]
    assert isinstance(navigation, dict)
    for command, raw_index in navigation.items():
        assert navigate_putting_samples(plan, 3, command) == raw_index

    nearest = expected["nearest"]
    assert isinstance(nearest, dict)
    projected = [tuple(item) for item in nearest["projected"]]
    assert (
        nearest_putting_sample(
            projected,
            tuple(nearest["pointer"]),
            hit_radius_px=nearest["hit_radius_px"],
        )
        == nearest["raw_index"]
    )


def test_split_zero_has_no_false_skid_and_split_is_forced_into_plan() -> None:
    series = PuttingSampleSeries(
        path_x_m=tuple(float(index) for index in range(20)),
        path_y_m=(0.0,) * 20,
        speeds_mps=tuple(float(20 - index) for index in range(20)),
        times_s=tuple(index * 0.002 for index in range(20)),
        skid_end_index=0,
    )
    plan = plan_putting_samples(series, cap=5)
    assert all(plan.raw_sample(index).phase == "pure-roll" for index in range(20))
    assert plan.skid_polyline_indices == ()
    assert plan.pure_roll_polyline_indices[0] == 0

    split_plan = plan_putting_samples(
        PuttingSampleSeries(
            path_x_m=series.path_x_m,
            path_y_m=series.path_y_m,
            speeds_mps=series.speeds_mps,
            times_s=series.times_s,
            skid_end_index=11,
        ),
        cap=5,
    )
    assert 11 in split_plan.displayed_raw_indices
    assert split_plan.skid_polyline_indices[-1] == 11
    assert split_plan.pure_roll_polyline_indices[0] == 11
    assert split_plan.raw_sample(11).phase == "pure-roll"


def test_fixed_plan_navigation_never_inserts_an_undisplayed_raw_sample() -> None:
    series = PuttingSampleSeries(
        path_x_m=tuple(float(index) for index in range(100)),
        path_y_m=(0.0,) * 100,
        speeds_mps=tuple(float(100 - index) for index in range(100)),
        times_s=tuple(index * 0.002 for index in range(100)),
        skid_end_index=17,
    )
    plan = plan_putting_samples(series, cap=8)
    before = plan.displayed_raw_indices
    undisplayed = next(index for index in range(100) if index not in before)
    assert navigate_putting_samples(plan, undisplayed, "next") == before[0]
    assert navigate_putting_samples(plan, undisplayed, "previous") == before[-1]
    assert plan.displayed_raw_indices == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("path_x_m", (0.0, float("nan"))),
        ("path_y_m", (0.0, float("inf"))),
        ("speeds_mps", (1.0, -0.1)),
        ("times_s", (0.0, 0.0)),
        ("times_s", (0.0, float("inf"))),
        ("skid_end_index", 2),
    ],
)
def test_malformed_series_fail_closed(field: str, value: object) -> None:
    values: dict[str, object] = {
        "path_x_m": (0.0, 1.0),
        "path_y_m": (0.0, 0.0),
        "speeds_mps": (1.0, 0.0),
        "times_s": (0.0, 0.002),
        "skid_end_index": 1,
    }
    values[field] = value
    with pytest.raises(ValueError):
        plan_putting_samples(PuttingSampleSeries(**values))


def test_real_legal_30001_sample_putt_plans_linearly_and_stays_bounded() -> None:
    result = simulate_putt(
        PuttLaunch(
            ball_speed_mps=0.2,
            launch_angle_deg=0.0,
            horizontal_speed_mps=0.2,
            spin_rad_s=0.0,
            effective_loft_deg=0.0,
        ),
        GreenConditions(stimp_ft=13.0, grade_percent=10.0, aspect_deg=0.0),
        40.0,
    )
    assert len(result.times_s) == 30_001
    series = PuttingSampleSeries.from_result(result)
    timings: list[float] = []
    for _ in range(3):
        started = time.perf_counter()
        plan = plan_putting_samples(series, cap=MAX_PUTTING_DISPLAY_SAMPLES)
        timings.append(time.perf_counter() - started)
    elapsed = min(timings)
    assert plan.raw_count == 30_001
    assert plan.displayed_count <= 1_024
    assert plan.displayed_raw_indices[0] == 0
    assert plan.displayed_raw_indices[-1] == 30_000
    assert result.skid_end_index in plan.displayed_raw_indices
    # The target is meaningful only for the isolated focused gate; the full
    # 14-worker GUI lane deliberately creates heavy CPU contention.
    if "PYTEST_XDIST_WORKER" not in os.environ:
        assert elapsed < 0.1


def test_nearest_rejects_outside_hit_radius_and_invalid_projection() -> None:
    assert nearest_putting_sample([(4, 10.0, 10.0)], (30.0, 30.0)) is None
    with pytest.raises(ValueError, match="projected"):
        nearest_putting_sample([(4, float("nan"), 10.0)], (10.0, 10.0))


def test_finite_raw_coordinates_cannot_overflow_derived_distance() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    series = PuttingSampleSeries(
        path_x_m=(maximum, -maximum),
        path_y_m=(0.0, 0.0),
        speeds_mps=(1.0, 0.0),
        times_s=(0.0, 0.002),
        skid_end_index=1,
    )
    with pytest.raises(ValueError, match="distance must remain finite"):
        plan_putting_samples(series)


def test_summary_must_match_exact_raw_evidence() -> None:
    from dataclasses import replace

    result = simulate_putt(
        PuttLaunch(2.0, 0.0, 2.0, 0.0, 0.0), GreenConditions(stimp_ft=10), 3.0
    )
    plan = plan_putting_samples(PuttingSampleSeries.from_result(result))
    validate_putting_result_summary(result, plan)
    with pytest.raises(ValueError, match="exact raw"):
        validate_putting_result_summary(replace(result, total_distance_m=1.0), plan)


def test_finite_extremes_cannot_collapse_the_display_envelope() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    series = PuttingSampleSeries(
        path_x_m=(0.0, 0.0),
        path_y_m=(0.0, maximum),
        speeds_mps=(maximum, 0.0),
        times_s=(0.0, 0.002),
        skid_end_index=1,
    )
    with pytest.raises(ValueError, match="display envelope"):
        plan_putting_samples(series)
