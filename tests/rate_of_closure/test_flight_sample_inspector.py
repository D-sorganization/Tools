"""Contracts for exact synchronized flight sample inspection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.flight_sample_inspector import (
    FlightSampleSelection,
    FlightSampleSeries,
    navigate_flight_samples,
    nearest_flight_sample,
    plan_flight_samples,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _golden() -> dict[str, object]:
    path = Path(__file__).parents[2] / (
        "src/rate_of_closure/web/src/model/__fixtures__/"
        "flight_sample_inspector_golden_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_python_owned_golden_pins_phase_navigation_and_pixel_tie() -> None:
    fixture = _golden()
    assert set(fixture) == {"schema_id", "schema_version", "series", "expected"}
    assert fixture["schema_id"] == "rate-of-closure/flight-sample-inspector-golden"
    assert fixture["schema_version"] == 1
    source, expected = fixture["series"], fixture["expected"]
    assert isinstance(source, dict) and isinstance(expected, dict)
    plan = plan_flight_samples(FlightSampleSeries(**source))
    assert plan.apex_raw_index == expected["apex_raw_index"]
    assert [sample.phase for sample in plan.samples] == expected["phases"]
    for command, raw_index in expected["navigation_from_3"].items():
        assert navigate_flight_samples(plan, 3, command) == raw_index
    nearest = expected["nearest"]
    selection = nearest_flight_sample(
        plan,
        [tuple(item) for item in nearest["projected"]],
        tuple(nearest["pointer"]),
        hit_radius_px=nearest["hit_radius_px"],
    )
    assert selection is not None
    assert {"cohort": selection.cohort, "raw_index": selection.raw_index} == nearest[
        "selection"
    ]


@pytest.mark.parametrize(
    ("times", "positions"),
    [
        ([0.0], [[0.0, 0.0, 0.0]]),
        ([0.0, 0.0], [[0.0, 0.0, 0.0]] * 2),
        ([0.0, 0.1], [[0.0, 0.0], [1.0, 0.0]]),
        ([0.0, 0.1], [[0.0, float("nan"), 0.0], [1.0, 0.0, 0.0]]),
    ],
)
def test_malformed_sample_evidence_fails_closed(
    times: object, positions: object
) -> None:
    with pytest.raises(ValueError):
        plan_flight_samples(FlightSampleSeries(times, positions))


def test_plan_snapshots_mutable_inputs_and_rejects_over_cap() -> None:
    times = [0.0, 0.1]
    positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    plan = plan_flight_samples(FlightSampleSeries(times, positions))
    times[1] = 9.0
    positions[1][0] = 9.0
    assert plan.raw_sample(1).time_s == 0.1
    assert plan.raw_sample(1).downrange_m == 1.0
    with pytest.raises(ValueError):
        FlightSampleSeries(range(1003), [[0.0, 0.0, 0.0]] * 1003)
    with pytest.raises(ValueError):
        FlightSampleSelection("current", -1)


def test_exact_cap_is_accepted_without_decimation() -> None:
    plan = plan_flight_samples(
        FlightSampleSeries(
            tuple(index * 0.001 for index in range(1002)),
            ((0.0, 0.0, 0.0),) * 1002,
        )
    )
    assert plan.raw_count == 1002


def test_picker_is_plan_bound_and_calm_comparison_is_not_selectable() -> None:
    plan = plan_flight_samples(
        FlightSampleSeries([0.0, 0.1], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )
    with pytest.raises(ValueError):
        nearest_flight_sample(
            plan, [("calm", 0, 0.0, 0.0), ("calm", 1, 1.0, 0.0)], (0.0, 0.0)
        )
    with pytest.raises(ValueError):
        nearest_flight_sample(plan, [("current", 0, 0.0, 0.0)], (0.0, 0.0))
    with pytest.raises(ValueError):
        nearest_flight_sample(
            plan,
            [("current", 0, 0.0, 0.0), ("current", 0, 1.0, 0.0)],
            (0.0, 0.0),
        )


def test_coincident_phase_precedence_is_honest_for_edge_trajectories() -> None:
    descending = plan_flight_samples(
        FlightSampleSeries([0.0, 1.0], [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0]])
    )
    rising = plan_flight_samples(
        FlightSampleSeries([0.0, 1.0], [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    )
    assert [sample.phase for sample in descending.samples] == ["launch", "landing"]
    assert [sample.phase for sample in rising.samples] == ["launch", "landing"]
    assert descending.apex_raw_index == 0
    assert rising.apex_raw_index == 1


@pytest.mark.parametrize(
    ("times", "positions"),
    [
        ([0.0, 10.001001], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ([0.0, 1.0], [[0.0, 0.0, 0.0], [10_000.001, 0.0, 0.0]]),
        ([0.0, 1.0], [[0.0, 0.0, 0.0], [float("1e308"), 0.0, 0.0]]),
    ],
)
def test_finite_but_unrenderable_evidence_fails_closed(
    times: object, positions: object
) -> None:
    with pytest.raises(ValueError):
        FlightSampleSeries(times, positions)
