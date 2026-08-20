"""Tests for descriptive launch-monitor performance adapter."""

from __future__ import annotations

import pandas as pd
import pytest

from rate_of_closure.launch_monitor_performance import (
    DispersionRequest,
    StrokesGainedRequest,
    TargetErrorRequest,
    TrendRequest,
    analyze_dispersion,
    analyze_session_trend,
    calculate_strokes_gained,
    calculate_target_error,
)


def test_dispersion_reports_units_and_directional_outcomes() -> None:
    result = analyze_dispersion(
        pd.DataFrame({"offline_m": [-9.144, 0.0, 4.572], "carry_m": [100, 101, 102]}),
        DispersionRequest("offline_m", "carry_m", "m", "m"),
    )
    assert result.unit == "yd"
    assert result.left_count == 1
    assert result.center_count == 1
    assert result.right_count == 1
    assert result.points[0].lateral_yards == pytest.approx(-10.0)
    assert result.points[2].carry_yards == pytest.approx(111.55, rel=1e-4)


def test_user_supplied_expected_strokes_sg_is_not_claimed_as_source_backed() -> None:
    frame = pd.DataFrame({"before": [3.2, 3.1], "after": [2.0, 1.8]})
    with pytest.raises(ValueError, match="source"):
        calculate_strokes_gained(frame, StrokesGainedRequest("before", "after", ""))
    result = calculate_strokes_gained(
        frame,
        StrokesGainedRequest(
            "before", "after", "https://datagolf.com/frequently-asked-questions"
        ),
    )
    assert result.values == pytest.approx((0.2, 0.3))
    assert result.mean == pytest.approx(0.25)
    assert result.unit == "strokes"
    assert result.metric_name == "user_supplied_expected_strokes_sg"
    assert "did not reproduce or validate" in result.formula


def test_distance_only_metric_is_named_target_error_not_strokes_gained() -> None:
    result = calculate_target_error(
        pd.DataFrame({"carry": [150.0], "offline": [12.0]}),
        TargetErrorRequest("carry", "offline", "yd", "yd", 160.0),
    )
    assert result.metric_name == "radial_target_error"
    assert result.unit == "yd"
    assert result.values == pytest.approx(((10.0**2 + 12.0**2) ** 0.5,))


def test_trend_requires_attested_identity_and_explicit_consistent_order() -> None:
    frame = pd.DataFrame(
        {
            "player": ["p1", "p1", "p1", "p1"],
            "session": ["a", "a", "b", "b"],
            "session_order": [1, 1, 2, 2],
            "ball_speed": [100.0, 102.0, 104.0, 106.0],
        }
    )
    with pytest.raises(ValueError, match="attested"):
        analyze_session_trend(
            frame,
            TrendRequest(
                "ball_speed", "session", "session_order", "player", False, True
            ),
        )
    result = analyze_session_trend(
        frame,
        TrendRequest("ball_speed", "session", "session_order", "player", True, True),
    )
    assert [
        (point.session_id, point.mean, point.cumulative_mean) for point in result.points
    ] == [
        ("a", 101.0, 101.0),
        ("b", 105.0, 103.0),
    ]

    inconsistent = frame.copy()
    inconsistent.loc[1, "session_order"] = 3
    with pytest.raises(ValueError, match="one order"):
        analyze_session_trend(
            inconsistent,
            TrendRequest(
                "ball_speed", "session", "session_order", "player", True, True
            ),
        )


def test_pyqt_performance_workspace_exposes_fail_closed_parity(qtbot) -> None:  # type: ignore[no-untyped-def]
    from rate_of_closure.ui.pyqt6.launch_monitor_performance_workspace import (
        LaunchMonitorPerformanceWorkspace,
    )

    panel = LaunchMonitorPerformanceWorkspace()
    qtbot.addWidget(panel)
    panel.set_dataset(
        pd.DataFrame(
            {
                "player": ["p1", "p1", "p1"],
                "session": ["a", "a", "b"],
                "order": [1, 1, 2],
                "carry": [150, 155, 160],
                "lateral": [-10, 5, 8],
                "speed": [100, 102, 106],
            }
        ),
        "test.csv",
    )
    panel.carry_combo.setCurrentText("carry")
    panel.lateral_combo.setCurrentText("lateral")
    dispersion, proxy = panel.run_dispersion()

    assert dispersion.left_count == 1
    assert dispersion.right_count == 2
    assert proxy.metric_name == "radial_target_error"
    assert "yd" in panel.dispersion_status.text()
    assert not panel.trend_button.isEnabled()
    assert "Unavailable" in panel.strokes_status.text()
