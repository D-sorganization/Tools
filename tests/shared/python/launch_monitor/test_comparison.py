"""Canonical monitor-comparison tests (ADR-0046 G1 step P4).

The first case is UpstreamDrift's
``tests/unit/launch_monitor/test_analysis.py::test_matched_monitor_comparison_recovers_bias_and_slope``,
travelling verbatim with the module it exercises. The remaining cases pin the
refusals and the descriptive-mode warning that
:mod:`shared.python.launch_monitor.comparison` already performs and its
docstring already documents — ``CLAUDE.md``'s design-by-contract rule asks
every ported public entry point to have them pinned.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from shared.python.launch_monitor.comparison import compare_monitors

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _matched_frame(pairs: int = 30) -> pd.DataFrame:
    """Return the paired two-monitor frame the ported case builds."""
    x = np.linspace(30.0, 50.0, pairs)
    return pd.DataFrame(
        {
            "match_id": np.repeat(np.arange(pairs), 2),
            "monitor_vendor": np.tile(["A", "B"], pairs),
            "ball_speed": np.column_stack([x, 1.03 * x + 0.8]).ravel(),
        }
    )


def test_matched_monitor_comparison_recovers_bias_and_slope() -> None:
    """Ported verbatim from UpstreamDrift's ``test_analysis.py``."""
    frame = _matched_frame(30)
    result = compare_monitors(
        frame,
        metric="ball_speed",
        monitor_column="monitor_vendor",
        match_column="match_id",
        reference_monitor="A",
    )
    comparison = result.pairwise[0]
    assert comparison.matched is True
    assert comparison.slope == pytest.approx(1.03, rel=0.01)
    assert comparison.intercept == pytest.approx(0.8, rel=0.1)
    assert comparison.mean_bias > 1.0


def test_matched_comparison_reports_limits_of_agreement() -> None:
    """The matched arm is Bland-Altman: bias plus 95% limits of agreement."""
    result = compare_monitors(
        _matched_frame(30),
        metric="ball_speed",
        monitor_column="monitor_vendor",
        match_column="match_id",
        reference_monitor="A",
    )
    comparison = result.pairwise[0]
    assert comparison.lower_limit == pytest.approx(
        comparison.mean_bias - 1.96 * comparison.standard_deviation_bias, rel=1e-12
    )
    assert comparison.upper_limit == pytest.approx(
        comparison.mean_bias + 1.96 * comparison.standard_deviation_bias, rel=1e-12
    )
    assert comparison.warning is None
    assert {summary.monitor for summary in result.summaries} == {"A", "B"}


def test_unmatched_comparison_refuses_to_claim_agreement(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Without a match key only the distributions are comparable.

    Every agreement statistic comes back ``nan`` and the result carries the
    confounding warning, so a descriptive gap can never be read as a device
    bias. The ``correlation`` slot instead carries the pooled-SD effect size.
    """
    result = compare_monitors(shots(80), metric="ball_speed")
    comparison = result.pairwise[0]
    assert comparison.matched is False
    assert np.isnan(comparison.standard_deviation_bias)
    assert np.isnan(comparison.lower_limit)
    assert np.isnan(comparison.upper_limit)
    assert np.isnan(comparison.slope)
    assert np.isnan(comparison.intercept)
    assert comparison.warning is not None
    assert "confounded" in comparison.warning
    assert np.isfinite(comparison.correlation)


def test_comparison_names_the_columns_it_cannot_find(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """A missing column is refused by name, never silently coerced away."""
    frame = shots(20).drop(columns=["monitor_vendor"])
    with pytest.raises(ValueError, match=r"Columns not present"):
        compare_monitors(frame, metric="ball_speed")


def test_comparison_requires_two_monitors(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """One monitor is not a comparison."""
    frame = shots(20).assign(monitor_vendor="TrackMan")
    with pytest.raises(ValueError, match=r"At least two monitors are required"):
        compare_monitors(frame, metric="ball_speed")


def test_comparison_refuses_an_absent_reference_monitor(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """The reference must be one of the monitors actually present."""
    with pytest.raises(ValueError, match=r"Reference monitor not present"):
        compare_monitors(shots(20), metric="ball_speed", reference_monitor="Foresight")


def test_matched_comparison_requires_three_matched_pairs() -> None:
    """A regression through two matched points is not evidence."""
    frame = _matched_frame(2)
    with pytest.raises(ValueError, match=r"At least three matched pairs"):
        compare_monitors(
            frame,
            metric="ball_speed",
            monitor_column="monitor_vendor",
            match_column="match_id",
            reference_monitor="A",
        )
