"""Python authority tests for bounded launch-monitor linked scatter."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_analysis import numeric_columns
from rate_of_closure.launch_monitor_linked_scatter import (
    navigate_linked_scatter,
    plan_linked_scatter,
    project_plot_axis,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _golden() -> dict[str, object]:
    path = (
        Path(__file__).parents[2]
        / "src/rate_of_closure/web/src/model/__fixtures__"
        / "launch_monitor_linked_scatter_golden_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_python_owned_golden_pins_filter_decimation_selection_and_navigation() -> None:
    fixture = _golden()
    rows = fixture["rows"]
    expected = fixture["expected"]
    assert isinstance(rows, list) and isinstance(expected, dict)
    plain = plan_linked_scatter(rows, "x", "y", cap=4)
    selected = plan_linked_scatter(rows, "x", "y", cap=4, selected_raw_index=6)

    assert (plain.raw_count, plain.finite_count) == (10, 7)
    assert [point.raw_index for point in plain.points] == expected[
        "unselected_raw_indices"
    ]
    assert [point.raw_index for point in selected.points] == expected[
        "selected_raw_indices"
    ]
    navigation = expected["navigation_from_selected"]
    assert isinstance(navigation, dict)
    for command, raw_index in navigation.items():
        assert navigate_linked_scatter(selected, 6, command) == raw_index
    assert selected.points[2].shot_id == "six"


def test_selected_nonfinite_row_clears_without_changing_raw_records() -> None:
    rows = [{"x": 1.0, "y": 2.0}, {"x": None, "y": 3.0}]
    before = [dict(row) for row in rows]
    plan = plan_linked_scatter(rows, "x", "y", selected_raw_index=1)
    assert plan.selected_raw_index is None
    assert rows == before


def test_100k_rows_remain_bounded_and_fast() -> None:
    rows = [{"x": index, "y": index % 97} for index in range(100_000)]
    started = time.perf_counter()
    plan = plan_linked_scatter(rows, "x", "y", selected_raw_index=50_001)
    elapsed = time.perf_counter() - started
    assert plan.displayed_count == 2_000
    assert any(point.raw_index == 50_001 for point in plan.points)
    # A coarse complexity guard, not a performance SLA. What it exists to catch
    # is an accidental super-linear pass over the rows, which at 100k inputs
    # costs minutes rather than fractions of a second -- so any ceiling in this
    # range catches it equally well. The previous 0.5 s budget was tight enough
    # that ordinary CI contention tripped it (main failed at 0.5146 s once the
    # suite began running under xdist, #4548), reporting a scheduling accident
    # as a code regression. Assert the bound that actually distinguishes the
    # two, and let `displayed_count` above carry the deterministic contract.
    assert elapsed < 5.0


def test_numeric_string_grammar_rejects_hex_and_accepts_decimal_exponents() -> None:
    grammar = _golden()["numeric_grammar"]
    assert isinstance(grammar, dict)
    values = grammar["values"]
    assert isinstance(values, list)
    rows = [{"x": value, "y": index + 1} for index, value in enumerate(values)]
    plan = plan_linked_scatter(rows, "x", "y")
    assert [point.raw_index for point in plan.points] == grammar["finite_indices"]
    assert numeric_columns(pd.DataFrame(rows)) == ["x", "y"]


@pytest.mark.parametrize("cap", [True, 1, 2001, 2.5])
def test_invalid_caps_fail_closed(cap: object) -> None:
    with pytest.raises(ValueError, match="cap"):
        plan_linked_scatter([{"x": 1, "y": 2}], "x", "y", cap=cap)


def test_python_owned_golden_pins_overflow_safe_plot_projection() -> None:
    cases = _golden()["plot_projection_cases"]
    assert isinstance(cases, dict)
    for case in cases.values():
        assert isinstance(case, dict)
        projection = project_plot_axis(case["values"])
        assert projection.coordinates == tuple(case["expected"])
        assert all(-1 <= value <= 1 for value in projection.coordinates)
