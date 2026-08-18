"""Cross-runtime exact-evidence contract for managed plot inspection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rate_of_closure.plot_point_inspector import (
    MAX_PLOT_SAMPLES,
    MAX_PLOT_VERTICES,
    HistogramSelection,
    SeriesSelection,
    histogram_bin_at_data,
    navigate_plot_selection,
    nearest_series_point,
    plan_plot_inspection,
)

pytestmark = pytest.mark.unit

_GOLDEN = json.loads(
    (
        Path(__file__).parents[2] / "src/rate_of_closure/web/src/model/__fixtures__/"
        "plot_point_inspector_golden_v1.json"
    ).read_text(encoding="utf-8")
)


def _selection(value: dict[str, object] | None):
    if value is None:
        return None
    if value["kind"] == "series":
        return SeriesSelection(int(value["series_index"]), int(value["raw_index"]))
    return HistogramSelection(int(value["bin_index"]))


def test_shared_series_plan_pick_and_navigation() -> None:
    case = _GOLDEN["series"]
    plan = plan_plot_inspection(case["kind"], case["x"], case["series"])
    assert plan.raw_count == 4
    assert tuple(series.label for series in plan.series) == ("Alpha", "Beta")
    assert nearest_series_point(plan, case["projected"], case["tie_pointer"]) == (
        _selection(case["tie_selection"])
    )
    assert nearest_series_point(plan, case["projected"], [0.0, 10.0]) == (
        SeriesSelection(1, 0)
    )
    assert nearest_series_point(plan, case["projected"], [0.0, 22.1]) is None
    for current, command, expected in case["navigation"]:
        assert navigate_plot_selection(
            plan, _selection(current), command
        ) == _selection(expected)


def test_shared_histogram_bins_and_navigation() -> None:
    case = _GOLDEN["histogram"]
    plan = plan_plot_inspection("histogram", case["x"], [])
    assert [(item.index, item.count) for item in plan.bins if item.count] == [
        tuple(item) for item in case["nonzero_bins"]
    ]
    assert histogram_bin_at_data(plan, 1.0, 1.0) == HistogramSelection(5)
    assert histogram_bin_at_data(plan, 1.0, 2.0) is None
    for current, command, expected in case["navigation"]:
        assert navigate_plot_selection(
            plan, _selection(current), command
        ) == _selection(expected)


def test_planner_rejects_resource_and_numeric_forgery_before_copy() -> None:
    class Oversized:
        traversed = False

        def __len__(self) -> int:
            return MAX_PLOT_SAMPLES + 1

        def __iter__(self):  # type: ignore[no-untyped-def]
            self.traversed = True
            yield from ()

    oversized = Oversized()
    with pytest.raises(ValueError, match="samples"):
        plan_plot_inspection("histogram", oversized, [])
    assert not oversized.traversed
    with pytest.raises(ValueError, match="finite bounded"):
        plan_plot_inspection(
            "line",
            [0.0, float("inf")],
            [{"label": "Y", "values": [1, 2]}],
        )
    x = list(range(MAX_PLOT_VERTICES // 2 + 1))
    with pytest.raises(ValueError, match="vertices"):
        plan_plot_inspection(
            "line",
            x,
            [{"label": "A", "values": x}, {"label": "B", "values": x}],
        )
