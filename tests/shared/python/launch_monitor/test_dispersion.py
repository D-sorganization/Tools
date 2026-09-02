"""Canonical dispersion tests (ADR-0046 G1 step P1).

The first case is the **dispersion half** of UpstreamDrift's
``tests/unit/launch_monitor/test_analysis.py::test_dispersion_and_longitudinal_trend_capture_change``,
split out per the port plan so the test travels with the module it exercises;
the trend half travels with :mod:`shared.python.launch_monitor.trends` in step
P3. The remaining cases pin the two refusals the module's docstring documents
as deliberate divergences from ``rate_of_closure``'s same-named function (G0
divergences D8 and D9).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from shared.python.launch_monitor.dispersion import analyze_dispersion

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_dispersion_captures_the_shot_pattern(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """Ported dispersion half of the UpstreamDrift combined trend/dispersion case."""
    frame = shots(80)
    dispersion = analyze_dispersion(
        frame, forward="carry_distance", lateral="lateral_carry"
    )
    assert dispersion.sample_count == 80
    assert dispersion.ellipse_major >= dispersion.ellipse_minor > 0
    assert dispersion.area_95 > 0


def test_dispersion_names_the_columns_it_cannot_find(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """A missing column is refused by name, never silently coerced away."""
    frame = shots(20).drop(columns=["lateral_carry"])
    with pytest.raises(ValueError, match=r"Dispersion columns not present"):
        analyze_dispersion(frame, forward="carry_distance", lateral="lateral_carry")


def test_dispersion_requires_three_complete_shots(
    shots: Callable[..., pd.DataFrame],
) -> None:
    """G0 divergence D8: this estimator needs a covariance, so three shots.

    ``rate_of_closure.launch_monitor_performance.analyze_dispersion`` accepts a
    single shot because it computes a 1-D lateral summary. The floors differ
    because the estimands differ; do not relax this one to match.
    """
    frame = shots(20).head(2)
    with pytest.raises(ValueError, match=r"At least three complete shots"):
        analyze_dispersion(frame, forward="carry_distance", lateral="lateral_carry")


def test_dispersion_declares_no_unit(shots: Callable[..., pd.DataFrame]) -> None:
    """G0 divergence D9: results come back in the frame's own unit.

    The result carries no ``unit`` field and the module performs no conversion,
    unlike the ``rate_of_closure`` function, which validates a declared unit and
    always reports yards.
    """
    frame = shots(40)
    result = analyze_dispersion(
        frame, forward="carry_distance", lateral="lateral_carry"
    )
    assert not hasattr(result, "unit")
    scaled = frame.assign(
        carry_distance=frame["carry_distance"] * 2.0,
        lateral_carry=frame["lateral_carry"] * 2.0,
    )
    doubled = analyze_dispersion(
        scaled, forward="carry_distance", lateral="lateral_carry"
    )
    assert doubled.radial_rmse == pytest.approx(2.0 * result.radial_rmse, rel=1e-12)
