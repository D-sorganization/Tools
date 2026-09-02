"""Canonical trend tests (ADR-0046 G1 step P3).

The first case is the **trend half** of UpstreamDrift's
``tests/unit/launch_monitor/test_analysis.py::test_dispersion_and_longitudinal_trend_capture_change``,
split out per the port plan (the dispersion half travels with
:mod:`shared.python.launch_monitor.dispersion` in step P1). The remaining
cases pin the ``TrendResult`` rename this step carries and the module's input
validation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from shared.python.launch_monitor import trends
from shared.python.launch_monitor.trends import TemporalTrendResult, analyze_trend

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _stepped(frame: pd.DataFrame) -> pd.DataFrame:
    """Add UpstreamDrift's mid-series +4.0 step to ``club_speed``."""
    stepped = frame.copy()
    stepped["club_speed"] += np.where(
        np.arange(len(frame)) >= len(frame) // 2, 4.0, 0.0
    )
    return stepped


def test_longitudinal_trend_captures_change(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Ported trend half of the UpstreamDrift combined trend/dispersion case."""
    trend_frame = _stepped(shots(80))
    trend = analyze_trend(
        trend_frame,
        metric="club_speed",
        time_column="captured_at",
        rolling_window=10,
    )
    assert trend.slope_per_day > 0
    assert trend.change_candidates
    assert trend.latest_mean > trend.earliest_mean


def test_trend_result_is_renamed_and_trendresult_is_not_bound(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """The P3 rename: no ``TrendResult`` here, not even as an alias.

    ``rate_of_closure.launch_monitor_performance`` exports a ``TrendResult``
    that computes cumulative session-ordinal means, a different estimand from
    this module's per-calendar-day slope. Binding the old name here — however
    convenient for Stage 2's import rewrite — would let the two merge silently
    in any namespace that saw both. A stale import must fail loudly instead.
    """
    result = analyze_trend(_stepped(shots(80)), metric="club_speed")
    assert isinstance(result, TemporalTrendResult)
    assert not hasattr(trends, "TrendResult")
    assert "TrendResult" not in trends.__all__


def test_trend_reports_time_based_not_ordinal_slope(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """The estimand is per day, so halving the elapsed span doubles the slope.

    This is precisely what the ``rate_of_closure`` twin cannot express: it
    orders observations by session and never sees elapsed time.
    """
    frame = _stepped(shots(80))
    daily = analyze_trend(frame, metric="club_speed", time_column="captured_at")
    twice_as_fast = frame.assign(
        captured_at=pd.date_range("2026-01-01", periods=len(frame), freq="12h")
    )
    compressed = analyze_trend(
        twice_as_fast, metric="club_speed", time_column="captured_at"
    )
    assert compressed.slope_per_day == pytest.approx(
        2.0 * daily.slope_per_day, rel=1e-9
    )
    assert compressed.sample_count == daily.sample_count


def test_trend_requires_a_rolling_window_of_at_least_three(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """A two-point window cannot support a rolling standard deviation."""
    with pytest.raises(ValueError, match=r"rolling_window must be at least three"):
        analyze_trend(shots(20), metric="club_speed", rolling_window=2)


def test_trend_names_the_columns_it_cannot_find(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """A missing metric or time column is refused by name."""
    with pytest.raises(ValueError, match=r"Trend columns not present"):
        analyze_trend(shots(20), metric="spin_rate")


def test_trend_requires_enough_complete_observations(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Fewer than ``max(6, rolling_window)`` complete rows is refused."""
    with pytest.raises(ValueError, match=r"Insufficient complete observations"):
        analyze_trend(shots(20).head(5), metric="club_speed", rolling_window=3)
